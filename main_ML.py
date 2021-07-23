import argparse
import os
import shutil
import time
import sys
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import GCA as models
from utils.tools_mML import *
from tqdm import tqdm
from loss_ML import *

parser = argparse.ArgumentParser(description='Attribute Framework')
parser.add_argument('--experiment', default='voc', type=str, help='(default=%(default)s)')
parser.add_argument('--approach', default='BNincept', type=str, help='(default=%(default)s)')
parser.add_argument('--epochs', default=90, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='(default=%(32)d)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--optimizer', default='adam', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--start-epoch', default=0, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--resume', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--decay_epoch', default=(40,70), type=eval, required=False, help='(default=%(default)d)')
parser.add_argument('--evaluate', default = False, help='evaluate model on validation set')

def get_config():
    config = {
        "alphaD": 0.05, ###  0.05
        "info": "[MLGCML]",
        "resize_size": 256,
        "crop_size": 256,
        "batch_size": 50,
        "alpha": 0.1, 
        "belta":5, 
        "gama":0.1,   ###  10 for the former, 0.1 for the later
        
        "dataset": "cifar10",
        # "dataset": "PID",
        # "dataset": "nuswide_21",
        # "dataset":"coco",

        "save_path": "savemML/CIF",
        # "save_path": "savemML/PID",
        # "save_path": "savemML/NUS",
        # "save_path": "savemML/CO",
        "epoch": 90,
        "evaluate_freq": 5,
        "GPU": True,

        "bit_list": [48],
    }
    if config["dataset"] == "cifar10":
        config["topK"] = 5000     ##   should be 5k
        config["n_class"] = 10
        
    elif config["dataset"] == "PID":
        config["topK"] = 22400      ##   should be 5k
        config["n_class"] = 14
        
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
        
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    config["data_path"] = "./data/" + config["dataset"] + "/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config
# Seed
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available(): torch.cuda.manual_seed(1)
else: print('[CUDA unavailable]'); sys.exit()
best_accu = 0
EPS = 1e-12

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# torch.cuda.set_device(1)
#####################################################################################################


def main(config,args, bit,filetime):
    
    split_bit = bit//4        ##  most important
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    # create model
    model = models.__dict__[args.approach](pretrained=True, bits =split_bit, num_classes =  config["n_class"])

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use

    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_accu = checkpoint['best_accu']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    time = datetime.datetime.now()
    filetime = '%d%d%d'%( time.month, time.day, time.hour)

    # if args.evaluate:
    #     if args.resume:
    #         print (args.resume)
    #         epoch = int(args.resume.split('/')[-1].split('.')[0])
    #     else:
    #         epoch = 0
            
    #     test_ml(val_loader, model, attr_num, description,epoch, filetime)
    #     return

    train(train_loader, test_loader, dataset_loader, num_train, num_test, model,  optimizer,config, args,split_bit, bit,filetime)
    


def train(train_loader, test_loader, dataset_loader, num_train, num_test, model, optimizer, config, args, split_bit, bit,filetime):
    """Train for one epoch on the training set"""
    
    U_3 = torch.zeros(num_train, split_bit).float()
    U_4 = torch.zeros(num_train, split_bit).float()
    U_5 = torch.zeros(num_train, split_bit).float()
    U_m = torch.zeros(num_train, split_bit).float()
    L = torch.zeros(num_train, config["n_class"]).float()

    U_3 = U_3.cuda()
    U_4 = U_4.cuda()
    U_5 = U_5.cuda()
    U_m = U_m.cuda()
    L = L.cuda()
    Best_mAP = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.decay_epoch)
        model.train()
    
        for input, label, index in tqdm( train_loader):
    
            label = label.cuda(non_blocking=True).float()
            input = input.cuda(non_blocking=True)
            maxlabel = decode(label.clone())
            
            b_3,b_4,b_5, b_m , out_label= model(input)
            
            ####################################################
            cr_loss = cross_loss(out_label, maxlabel)
            # cr_loss = get_sigmoid_ce(out_label, label)     ##  for multi-label
            U_3[index, :] = b_3.data
            U_4[index, :] = b_4.data
            U_5[index, :] = b_5.data
            U_m[index, :] = b_m.data
            L[index, :] = label
            
            loss_3 = calc_loss(b_3, U_3, label, L, config)
            loss_4 = calc_loss(b_4, U_4, label, L, config)
            loss_5 = calc_loss(b_5, U_5, label, L, config)
            loss_m = calc_loss(b_m, U_m, label, L, config)
            
            loss_a = loss_3 + loss_4 + loss_5 + loss_m
            loss = 0.8*cr_loss +0.2*loss_a
            ##################################

            # cr_loss = cross_loss(out_label, maxlabel)
            # cr_loss = get_sigmoid_ce(out_label, label)     ##  for multi-label
            # loss_3 = hashing_loss_DY(b_3, label, config["alpha"], config["lamda"], config["gama"],  config["n_class"], bit)
            # loss_4= hashing_loss_DY(b_4, label, config["alpha"], config["lamda"], config["gama"],  config["n_class"], bit)
            # loss_5 = hashing_loss_DY(b_5, label, config["alpha"], config["lamda"], config["gama"],  config["n_class"], bit)
            # loss_m,= hashing_loss_DY(b_m, label, config["alpha"], config["lamda"], config["gama"],  config["n_class"], bit)
            
            # loss = loss_3 + loss_4 + loss_5 + loss_m
            # loss = 0.8*cr_loss +0.2* loss_a
            # #############################
    
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
        if (epoch + 1) % config["evaluate_freq"] == 0:
        # if (epoch + 1) % 1 == 0:
            model.eval()
            tst_binary, tst_label = compute_result(test_loader, model, usegpu=config["GPU"])

            # trn_binary, trn_label = compute_result(train_loader, model, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, model, usegpu=config["GPU"])

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),config["topK"])

            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            if mAP > Best_mAP:
                Best_mAP = mAP
                
            if "save_path" in config:
                newpath = config["save_path"] + '/%s'%filetime
                
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                print("save in ", newpath)
                # np.save(os.path.join(newpath, config["dataset"] + "-" + str(bit) + "-" + str(mAP) + "-" + "trn_binary.npy"),                        trn_binary.numpy())
                torch.save(model.state_dict(),
                           os.path.join(newpath, config["dataset"] + "-" + str(bit)+ "-" + str(mAP) + "-model.pt"))
            model.train()
    print("bit:%d,Best MAP:%.3f" % (bit, Best_mAP))


def adjust_learning_rate(optimizer, epoch, decay_epoch):
    lr = args.lr
    #### warm-up
    # if epoch <= 5:
    #     lr = lr * ((epoch+1)/5)
            
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    timenow = datetime.datetime.now()
    filetime = '%d%d%d'%( timenow.month, timenow.day, timenow.hour)
    global args
    args = parser.parse_args()
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        main(config, args, bit,filetime)
