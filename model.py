import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy.ndimage import gaussian_filter
import numpy as np

class ImgNet(nn.Module):
    def __init__(self, img_feat_len,code_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear( img_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):

        x = x.view(x.size(0), -1).float()
        feat1 = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat1))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self,  txt_feat_len,code_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc_encode = nn.Linear(4096, code_len)

        self.alpha = 1.0
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.normal(self.fc_encode.weight, mean=0.0, std=0.3)

        # self.init_weights()   #coco时屏蔽掉  25k nus时保留
    def init_weights(self):

        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc_encode.weight)

        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc_encode.bias is not None:
            init.constant_(self.fc_encode.bias, 0)
    def forward(self, x):
        feat = self.relu(self.fc1(x))
        hid = self.fc_encode(self.dropout(feat))
        code = torch.tanh(self.alpha * hid)

        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
#
def cal_similarityTag(config,img,txt,tag,knn_number = 3500,scale = 4000,a2 = 0.7):

    F_I = F.normalize(img)
    F_T = F.normalize(txt)

    sim=tag.mm(tag.t())
    S_tag=1/(1+torch.exp(-sim))
    S_tag=S_tag * 2 - 1


    S_I = F_I.mm(F_I.t())
    S_T= F_T.mm(F_T.t())


    S_img_txt=S_I*config.lamda1+ S_T*(1-config.lamda1)


    bs =S_img_txt.size(0)

    mask = torch.tril(torch.ones(bs, bs, dtype=torch.bool), diagonal=-1)
    non_diag_elements = S_img_txt[mask]

    mean_value = non_diag_elements.mean().item()
    std_value = non_diag_elements.std().item()

    left =  mean_value -std_value*config.L1
    right = mean_value +std_value*config.L2

    S_1=S_img_txt.clone()
    S_2=S_tag.clone()

    condition1 =(S_2 ==0 )& (S_1 > left)
    S_tag[condition1] = S_1[condition1] * config.α2

    condition2 = (S_1 < right) & (S_1 > left)
    S_img_txt[condition2]=config.α2 * S_2[condition2] * S_img_txt[condition2]

    S_img_txt[(S_2 == 0) & (S_1 < left )] = 0
    S_img_txt[(S_2 > 0) & (S_1 > right)] = 1

    pro = S_tag*config.lamda2+  S_img_txt*(1-config.lamda2)

    size = img.size(0)
    top_size = knn_number
    m, n1 = pro.sort()
    pro[torch.arange(size).view(-1, 1).repeat(1, top_size).view(-1), n1[:, :top_size].contiguous().view(-1)] = 0.
    pro[torch.arange(size).view(-1), n1[:, -1:].contiguous().view(-1)] = 0.
    pro = pro / pro.sum(1).view(-1, 1)
    pro_dis = pro.mm(pro.t())
    pro_dis = pro_dis * scale
    # pdb.set_trace()
    S = pro_dis * a2
    S = S * 2.0 - 1

    return S, S_1, S_2


def High_sample(S_s,S_t,config):

    delta = (S_s - S_t).abs()

    A = 1 + torch.exp(-delta) * config.gamma

    W = torch.where(delta < config.epsilon, A, 1.0)

    return W
