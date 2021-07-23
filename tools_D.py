import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable


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
    for img, targets, _ in tqdm(dataloader):
    # for img, targets, batch_idx in dataloader:
        
        clses.append(targets)
        if usegpu:
            outputs, _ = net(img.cuda())
            bs.append(outputs.data.cpu())
            # bs.append((net(img)).data.cpu())
        else:
            outputs, _ = net(img)
            bs.append(outputs.data.cpu())
            # bs.append((net(img)).data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)

def binary_output(dataloader, net):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
        
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()
    
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for inputs, targets, _ in tqdm(dataloader):
            # print (batch_idx)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs, _ = net(inputs)
            # print (outputs)
            full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)
            # print (torch.sign(full_batch_output))
        return torch.sign(full_batch_output), full_batch_label    ###  round

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
        tsum = np.sum(tgnd).astype(int)  ### 
        
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)    ###  等差数列 1-tsum

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def calc_map(rB, qB, retrievalL, queryL ):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    map = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in tqdm (range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        
        tsum = np.sum(gnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map

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
    sum_tp_R = np.zeros(trainset_len)
    
    for i in range(query_times):
        # print('Query ', i+1)
        query_label = tst_label[i,:]
        query_binary = tst_binary[i,:]
        # query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        # print(query_result.shape)
        
        query_result = CalcHammingDist(query_binary, trn_binary)
        
        sort_indices = np.argsort(query_result)
        # buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
        
        buffer_yes = (np.dot(query_label, trn_label[sort_indices].transpose()) > 0).astype(int)
        # print(buffer_yes.shape)
        # print(buffer_yes)
        
        exist_yes = sum(buffer_yes)
        # print(exist_yes)
        
        # exist_yes_sum = exist_yes_sum + exist_yes
        
        P = np.cumsum(buffer_yes) / Ns
        # print(P.shape)
        
        # print(np.max(query_result))
        # print(np.where(np.sort(query_result)>0)[0].shape)
        
        # precision_radius[i] = P[np.where(np.sort(query_result)>2)[0][0]-1]
        
        # AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        
        count = np.linspace(1, exist_yes, exist_yes)
        tindex = np.asarray(np.where(buffer_yes == 1)) + 1.0
        AP[i] = np.mean(count / (tindex))

        # print (AP[i])
        sum_tp = sum_tp + np.cumsum(buffer_yes)
        
        sum_tp_R = sum_tp_R + np.cumsum(buffer_yes)/exist_yes
        
    precision_at_k = sum_tp / Ns / query_times
    # recall_at_k = sum_tp / (trainset_len/classes) /query_times
    recall_at_k = sum_tp_R / query_times
    
    index = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    print('recall at k:', recall_at_k[index])
    
    # np.save(config["save_path"]+'/precision_at_k_%s'%config["info"]+'_%d'%bit, precision_at_k)
    # np.save(config["save_path"]+'/recall_at_k_%s'%config["info"]+'_%d'%bit, recall_at_k)
    
    
    # print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)
    return map


