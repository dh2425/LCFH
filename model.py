import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
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





def cal_similarityTag(config,img,txt,tag,knn_number = 3500,scale = 4000,a2 = 0.7):
    tag_sim = tag.mm(tag.t())


    F_I = F.normalize(img)
    F_T = F.normalize(txt)
    S_I = F_I.mm(F_I.t())
    S_T= F_T.mm(F_T.t())


    S_img_txt=S_I*config.lamda1+ S_T*(1-config.lamda1)

    bs =S_img_txt.size(0)

    mask = torch.tril(torch.ones(bs, bs, dtype=torch.bool), diagonal=-1)


    S_T=S_img_txt.clone()
    S_S = adj_matrix(tag_sim.clone())


    non_diag_v = S_I[mask]
    non_diag_t = S_T[mask]


    printPlt(non_diag_v ,non_diag_t)


    mean_value_v = non_diag_v .mean().item()
    std_value_v = non_diag_v .std().item()

    left_v = mean_value_v - std_value_v  * config.eta_1
    right_v = mean_value_v + std_value_v  * config.eta_2

    mean_value_t = non_diag_t.mean().item()
    std_value_t = non_diag_t.std().item()
    left_t = mean_value_t - std_value_t * config.eta_1
    right_t = mean_value_t + std_value_t * config.eta_2


    weight_enhance = torch.sigmoid(((S_I - right_v) + (S_T - right_t))/2)
    condition_enhance = (tag_sim == 0) & (S_I > right_v) & (S_T > right_t)
    # print(sum(sum(condition_enhance)))

    weight_enhance2 = torch.sigmoid(((left_v - S_I) + (left_t - S_T)/2))
    condition_enhance2 = (tag_sim > 1) & (S_T < left_t) & (S_I < left_v)
    # print(sum(sum(condition_enhance2)))

    S_tag = adj_matrix(tag_sim)
    S_tag[condition_enhance] = S_img_txt[condition_enhance]*weight_enhance[condition_enhance]
    S_tag[condition_enhance2]=S_tag[condition_enhance2]*(1-weight_enhance2[condition_enhance2])+S_img_txt[condition_enhance2]*weight_enhance2[condition_enhance2]


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
    return S, S_T, S_S

def printPlt(non_diag_v, non_diag_t):
    import matplotlib.pyplot as plt
    non_diag_v = non_diag_v.cpu().numpy()
    non_diag_t = non_diag_t.cpu().numpy()

    # 设置边框样式
    border_color = 'black'  # 边框颜色
    border_width = 1.5  # 边框宽度

    # 第一个单独的图表 - non_diag_v
    plt.figure(figsize=(10, 6))  # 单个图表的大小
    plt.hist(non_diag_v, bins=500, color='#006698', alpha=1)
    plt.grid(axis='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7, zorder=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 设置所有spines可见并自定义样式
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    plt.tight_layout()
    try:
        plt.savefig(r'png\image.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"保存失败: {e}")
    plt.show()

    # 第二个单独的图表 - non_diag_t
    plt.figure(figsize=(10, 6))  # 单个图表的大小
    plt.hist(non_diag_t, bins=500, color='#56885B', alpha=1)
    plt.grid(axis='both', linestyle='--', linewidth=0.5, color='lightgray', alpha=0.7, zorder=0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 设置所有spines可见并自定义样式
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(border_width)

    plt.tight_layout()
    try:
        plt.savefig(r'png\caption.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"保存失败: {e}")
    plt.show()

def adj_matrix(sim):
    S_e=1/(1+torch.exp(-sim))
    S=S_e * 2 - 1
    return S

def High_sample(S_s,S_t,k=10):
    gc_s = smiK(S_s, k=k)
    gc_t = smiK(S_t, k=k)
    gc = gc_s * gc_t
    w = gc.sum(dim=-1)
    w = F.normalize(w, dim=-1)
    weight_S = w
    weight = weight_S.unsqueeze(1) * weight_S.unsqueeze(0)

    # weight_matrix = 1 / (1 + torch.exp(-weight * 10))
    # weight_matrix = 1 / (1 + torch.exp(-weight * w.size(0)))
    weight_matrix = 1 / (1 + torch.exp(-weight))
    # T1=weight_matrix.min()
    # T2 = weight_matrix.max()
    return weight_matrix

def smiK(emb, k):
    k=k+1
    if k >emb.size(0):
        k = emb.size(0)

    # emb=F.normalize(emb,dim=1)
    # emb = torch.softmax(emb / 1, dim=-1)*emb.size(0)
    topk_values, topk_indices = torch.topk(emb, k, dim=1)
    # 创建一个全零张量
    result = torch.zeros_like(emb)
    # 将每行的最大 k 个元素填充到对应位置
    result.scatter_(1, topk_indices, topk_values)
    mask = 1 - torch.eye(emb.size(0)).to(emb.device)
    result = result * mask

    return result
