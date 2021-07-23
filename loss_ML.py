import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from util.distance import distance

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def decode(datum):
    return torch.argmax(datum, dim = 1)

def cross_loss(pred,target):
    criteria = nn.CrossEntropyLoss().cuda()
    loss = criteria(pred, target)
    return loss

def get_ce_sig_loss(pred,target):
    criteria = nn.BCELoss(pred,target).cuda()
    loss = criteria(pred, target)
    return loss

def get_sigmoid_ce(predictions,target):
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    # loss = nn.NLLLoss()
    predictions=m(predictions)
    output = loss(predictions, target.cuda())
    return output

###DPSH
def calc_loss_(x1, x2, y1, y2, config):
    s = (y1 @ y2.t() > 0).float()
    inner_product = x1 @ x2.t() * 0.5
    if config["GPU"]:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]).cuda())
    else:
        log_trick = torch.log(1 + torch.exp(-torch.abs(inner_product))) \
                    + torch.max(inner_product, torch.FloatTensor([0.]))
    loss = log_trick - s * inner_product
    loss1 = loss.mean()
    loss2 = config["alphaD"] * (x1 - x1.sign()).pow(2).mean()
    return loss1 + loss2

def calc_loss(x1, x2, y1, y2, config):
    s = (y1 @ y2.t() > 0).float()
    inner_product = x1 @ x2.t() * 0.5
    likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

    loss1 = likelihood_loss.mean()
    loss2 = config["alphaD"] * (x1 - x1.sign()).pow(2).mean()
    return loss1 + loss2

class DPSHLoss(torch.nn.Module):
    def __init__(self,num_train, config, bit):
        super(DPSHLoss, self).__init__()
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, config["n_class"]).float().cuda()

    def forward(self, u, y, ind, config, out_label):
        
        cr_loss = get_sigmoid_ce(out_label, y)
        
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5

        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        likelihood_loss = likelihood_loss.mean()

        quantization_loss = config["alpha"] * (u - u.sign()).pow(2).mean()
        
        dploss = likelihood_loss + quantization_loss
        sum_loss = 0.2*dploss + 0.8*cr_loss 
        return sum_loss


def same_loss(b_3,b_4,b_5,b_m,config):
    loss = config["alphaD"] * ( (b_3 - b_m.sign()).pow(2).mean()+(b_4 - b_m.sign()).pow(2).mean()+(b_5 - b_m.sign()).pow(2).mean())
    return loss
    
def hashing_loss_DY_Upd(NBcode,  x2, y1, y2, config,num_bit):
    
    ##### y1 is the current label
    loss_1 = calc_loss(NBcode, x2, y1, y2, config)
    
    ###  Nbcode is 32x32, y1 is 32x35, the first 32 is the batch size
    
    label_count = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(y1), dim=1)+ 1e-8),1)    ###32x1
    # print (label_count.shape)
    # norm_label = y1/torch.repeat(label_count,[1,args.config["n_class"]])        ###  cosin distance
    norm_label = y1/label_count.repeat(1,config["n_class"])        ###  cosin distance    ###32x35
    # print (norm_label.shape)
    
    w_label = torch.matmul(norm_label, norm_label.t())     ###  Sij   ### 32x32
    # print(w_label.shape)
    # w_label = torch.bmm(norm_label, norm_label.transpose(1,2))     ###  Sij
    ########################################
    #w_label = torch.cosine_similarity(y1, y1, dim=1)  ###  pytorch default function for cosin similarity
    
    semi_label = torch.where(w_label>0.99, w_label-w_label,w_label)    ### Mij in paper
    p2_distance = torch.matmul(NBcode, NBcode.t())     ##  NBcode is the binary code   x/(1 + abs(x))  -1 to 1   ##32x32
    # p2_distance = torch.bmm(NBcode, NBcode.transpose(1,2))     ##  NBcode is the binary code   x/(1 + abs(x))  -1 to 1
    # print (p2_distance.shape)
    
    scale_distance = config['alpha'] * p2_distance / num_bit      ### scale inner product, m is num_bit , belta is alpha in paper
    
    temp1 = torch.sigmoid(scale_distance)        ####sigmoid of scale inner product
    ###################
    # NBcode_N = torch.ge(NBcode, 0.5).float()
    # code_count = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(NBcode_N), dim=1)+ 1e-8),1)    ###32x1
    # norm_code = NBcode_N/code_count.repeat(1,num_bit)        ###  cosin distance    ###32x35
    # temp1 = torch.matmul(norm_code, norm_code.t())     ###  Sij  for NBcode  ### 32x32
    ###################
    loss = torch.where(semi_label<0.01, loss_1, config['gama']*torch.square(w_label-temp1))   ### L  32x32
    # print (loss.shape)
    
    # regularizer = torch.mean(torch.abs(torch.abs(NBcode) - 1))    ###  Q
    regularizer = torch.mean((torch.abs(NBcode) - 1)** 2)    ###  Q
    # print (regularizer)
    d_loss = (1-config['lamda'])*torch.mean(loss) + config['lamda'] * regularizer ###    alpha is lamda in paper
    # print(d_loss)
    
    return d_loss

class MLLoss_DY(torch.nn.Module):
    def __init__(self,num_train, config, bit):
        super(MLLoss_DY, self).__init__()
        self.num_bit = bit
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, config["n_class"]).float().cuda()

    def forward(self, u, y, ind, config, out_label):
        
        # cr_loss = get_sigmoid_ce(out_label, y)     ##  for multi-label
        
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = (y @ self.Y.t() > 0).float()
        inner_product = u @ self.U.t() * 0.5
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
        likelihood_loss = likelihood_loss.mean()
        quantization_loss = config["alphaD"] * (u - u.sign()).pow(2).mean()
        
        LossDP =  likelihood_loss 
        ###########
        
        label_count = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(y), dim=1)+ 1e-8),1)    ###32x1
        norm_label = y/label_count.repeat(1,config["n_class"])        ###  cosin distance    ###32x35
        w_label = torch.matmul(norm_label, norm_label.t())     ###  Sij   ### 32x32
                
        semi_label = torch.where(w_label>0.99, w_label-w_label,w_label)    ### Mij in paper
        
        # semi_l = torch.where(w_label>0.99,1, w_label)    ### Mij in paper
        # semi_label_ = torch.where(semi_label<0.01, 0, semi_l)    ### Mij in paper
        
        p2_distance = torch.matmul(u, u.t())     ##  u is the binary code   x/(1 + abs(x))  -1 to 1   ##32x32
        scale_distance = config['alpha'] * p2_distance / self.num_bit      ### scale inner product, m is num_bit , belta is alpha in paper

        temp1 = torch.sigmoid(scale_distance)        ####sigmoid of scale inner product
        ###################
        # u_o = torch.add(u, 1)
        # u_N = torch.ge( torch.div(u_o, 2), 0.5).float()
        # code_count = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(u_N), dim=1)+ 1e-8),1)    ###32x1
        # norm_code = u_N/code_count.repeat(1,self.num_bit)        ###  cosin distance    ###32x35
        # temp1 = torch.matmul(norm_code, norm_code.t())     ###  Sij  for u  ### 32x32
        ###################
        margin = int (0.5 *config["n_class"] )
        batch_size = y.shape[0]
        dist = distance(y)
        loss_c = w_label * dist + (1 - w_label) * torch.max(margin - dist, torch.zeros_like(dist))
        loss_cc = torch.sum(loss_c) / (batch_size*(batch_size-1))
        ######################
        # norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        # s = s / (norm + 0.00001)
        # mse_loss = (inner_product + self.num_bit - 2 * s * self.num_bit).pow(2)
        #########################
        # loss = torch.where(semi_label<0.01, LossDP, config['gama']*torch.square(w_label-temp1))   ### L  32x32
        # loss = torch.where(semi_label<0.01, loss_cc, LossDP)
        loss = torch.where(semi_label<0.01, loss_cc, (LossDP+config['gama']*torch.square(w_label-temp1)))
        sum_loss = torch.mean(loss) + quantization_loss

        return sum_loss

def hashing_loss_DY(NBcode, label, alpha, lamda, gama, num_class,num_bit):
    ###  Nbcode is 32x32, label is 32x35, the first 32 is the batch size
    
    label_count = torch.unsqueeze(torch.sqrt(torch.sum(torch.square(label), dim=1)+ 1e-8),1)    ###32x1
    # print (label_count.shape)
    # norm_label = label/torch.repeat(label_count,[1,args.num_class])        ###  cosin distance
    norm_label = label/label_count.repeat(1,num_class)        ###  cosin distance    ###32x35
    # print (norm_label.shape)
    
    w_label = torch.matmul(norm_label, norm_label.t())     ###  Sij   ### 32x32
    # print(w_label.shape)
    # w_label = torch.bmm(norm_label, norm_label.transpose(1,2))     ###  Sij
    ########################################
    #w_label = torch.cosine_similarity(label, label, dim=1)  ###  pytorch default function for cosin similarity
    
    semi_label = torch.where(w_label>0.99, w_label-w_label,w_label)    ### Mij in paper
    p2_distance = torch.matmul(NBcode, NBcode.t())     ##  NBcode is the binary code   x/(1 + abs(x))  -1 to 1   ##32x32
    # p2_distance = torch.bmm(NBcode, NBcode.transpose(1,2))     ##  NBcode is the binary code   x/(1 + abs(x))  -1 to 1
    # print (p2_distance.shape)
    
    scale_distance = (alpha/num_bit) * p2_distance      ### scale inner product, m is num_bit , belta is alpha in paper
    temp = torch.log(1+torch.exp(scale_distance))
    
    temp1 = torch.sigmoid(scale_distance)        ####sigmoid of scale inner product
    loss = torch.where(semi_label<0.01,(temp - w_label * scale_distance), gama* (torch.square(w_label-temp1)))   ### L  32x32
    # print (loss.shape)
    
    # regularizer = torch.mean(torch.abs(torch.abs(NBcode) - 1))    ###  Q
    regularizer = torch.mean((torch.abs(NBcode) - 1)** 2)    ###  Q
    # print (regularizer)
    d_loss = (1-lamda)*torch.mean(loss) + lamda * regularizer ###    alpha is lamda in paper
    # print(d_loss)
    
    return d_loss

class IDHNLoss(torch.nn.Module):
    def __init__(self,num_train, config, bit):
        super(IDHNLoss, self).__init__()
        self.q = bit
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, config["n_class"]).float().cuda()

    def forward(self, u, y, ind, config, out_label):
        u = u / (u.abs() + 1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = y @ self.Y.t()
        norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        s = s / (norm + 0.00001)

        M = (s > 0.99).float() + (s < 0.01).float()

        inner_product = (5/self.q) * u @ self.U.t()

        log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product

        mse_loss = (inner_product + self.q - 2 * s * self.q).pow(2)

        loss1 = (M * log_loss + 0.1 * (1 - M) * mse_loss).mean()
        loss2 = 0.1 * (u.abs() - 1).abs().mean()

        return loss1 + loss2


class ISDHLoss(torch.nn.Module):
    def __init__(self, num_train,config, bit):
        super(ISDHLoss, self).__init__()
        self.q = bit
        self.U = torch.zeros(num_train, bit).float().cuda()
        self.Y = torch.zeros(num_train, config["n_class"]).float().cuda()


    def forward(self, u, y, ind, config, out_label):
        u = u / (u.abs() + 1)
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        s = y @ self.Y.t()
        norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ self.Y.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        s = s / (norm + 0.00001)

        M = (s > 0.99).float() + (s < 0.01).float()

        inner_product = (5/self.q) * u @ self.U.t()

        log_loss = torch.log(1 + torch.exp(-inner_product.abs())) + inner_product.clamp(min=0) - s * inner_product
        mse_loss = (s - torch.sigmoid(inner_product)).pow(2)

        loss1 = (10 * M * log_loss + (1 - M) * mse_loss).mean()
        loss2 = 0.1 * (u.abs() - 1).abs().mean()

        return loss1 + loss2


def contrastive_loss(output, label, margin=24):    ###  margin = int (0.5 *args.class_num )
    '''contrastive loss
    - Deep Supervised Hashing for Fast Image Retrieval
    '''
    batch_size = output.shape[0]
    S =  torch.mm(label.float(), label.float().t())
    
    one = torch.ones_like(S)
    S = torch.where(S >0.9 , one, S)    ###   for multi-label retrieval
    
    dist = distance(output)
    loss_1 = S * dist + (1 - S) * torch.max(margin - dist, torch.zeros_like(dist))
    loss = torch.sum(loss_1) / (batch_size*(batch_size-1))
    return loss


def quantization_loss(output):
    loss = torch.mean((torch.abs(output) - 1) ** 2)
    return loss

def correlation_loss(output):
    loss = (output @ torch.ones(output.shape[1], 1)).sum()
    
    return loss