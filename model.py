import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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

        self.init_weights()
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

def cal_similarityTag(config,img,txt,tag,knn_number = 3500,scale = 4000,a2 = 0.7):

    img_F = F.normalize(img, dim=-1)
    tags_F = F.normalize(tag, dim=-1)


    cosine_similarity = torch.bmm(img_F.unsqueeze(1), tags_F.transpose(1, 2)).squeeze(1)

    bool_greater_than_threshold = cosine_similarity > config.tag_threshold


    _, max_indices = cosine_similarity.max(dim=1)
    bool_max_values = torch.zeros_like(cosine_similarity, dtype=torch.bool)
    bool_max_values.scatter_(1, max_indices.unsqueeze(1), True)

    bool_mask = bool_greater_than_threshold | bool_max_values
    mask = bool_mask.unsqueeze(-1).expand_as(tags_F)
    tags_mask = tag * mask
    tags_sum = tags_mask.sum(dim=1)
    is_true = torch.sum(bool_mask, dim=1)
    weights = is_true.unsqueeze(1).expand_as(tags_sum)
    tags_weights = tags_sum / weights

    F_tag= F.normalize(tags_weights, dim=-1)
    F_I = F.normalize(img)
    F_T = F.normalize(txt)

    S_tag=F_tag.mm(F_tag.t())
    S_I = F_I.mm(F_I.t())
    S_T= F_T.mm(F_T.t())

    pro =  S_tag*config.lamda1+ S_I*config.lamda2+ S_T*config.lamda3

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

    return S

