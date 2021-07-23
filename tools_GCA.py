import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip()]
    else:
        step = []
    return transforms.Compose([transforms.Resize([resize_size,resize_size])
                               # ,transforms.CenterCrop(crop_size)
                                ]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(), \
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set], \
                                                      batch_size=data_config[data_set]["batch_size"], \
                                                      shuffle=True, num_workers=0)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], len(dsets["train_set"]), len(
        dsets["test"])

def compute_result(dataloader, net, usegpu=True):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        if usegpu:
            b_3,b_4,b_5, b_m, label =  net(img.cuda())
            b = torch.cat( ( torch.cat( (torch.cat((b_3,b_4),1) ,b_5), 1), b_m) , 1)
            bs.append(b.data.cpu())
        else:
            b_3,b_4,b_5, b_m =  net(img)
            b = torch.cat( ( torch.cat( (torch.cat((b_3,b_4),1) ,b_5), 1) , b_m) , 1)
            bs.append(b.data.cpu())
            
    return torch.sign(torch.cat(bs)), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    classes = np.max(tst_label) + 1
    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]
            tst_sample_label = np.array([i]).repeat(100)
            continue
        else:
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(100)])
    query_times = tst_sample_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    
    for i in range(query_times):
        print('Query ', i+1)
        query_label = tst_sample_label[i]
        query_binary = tst_sample_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        # print(query_result.shape)
        
        sort_indices = np.argsort(query_result)
        buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
        
        P = np.cumsum(buffer_yes) / Ns
        print(np.max(query_result))
        # print(np.where(np.sort(query_result)>0)[0].shape)
        
        precision_radius[i] = P[np.where(np.sort(query_result)>2)[0][0]-1]
        
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        sum_tp = sum_tp + np.cumsum(buffer_yes)
        
    precision_at_k = sum_tp / Ns / query_times
    recall_at_k = sum_tp / (trainset_len/classes) /query_times
    
    index = [100, 200,300, 400,500, 600,700, 800,900, 1000,1100, 1200,1300, 1400,1500, 1600]
    # index = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    print('recall at k:', recall_at_k[index])
    # np.save(args.path+'/precision_at_k', precision_at_k)
    # np.save(args.path+'/recall_at_k', recall_at_k)
    
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)