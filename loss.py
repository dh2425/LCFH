import torch
import torch.nn.functional as F
import torch.nn as nn 

class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.device= device

    def forward(self, emb_i, emb_j):
        batch_size = emb_i.shape[0]
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(self.device)).float()
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss



def loss_W(H,S,W=None):
    if W is not None:
        sample_loss=(W*(H - S.detach())).pow(2)
        loss= sample_loss.sum()/ (W.sum())

        # 该方法如果选择k=0 则等于不使用该加权
        # sample_loss = (H - S.detach()).pow(2)
        # loss = (W * sample_loss).sum() / W.sum()

    else:
        loss=(H-S.detach()).pow(2).mean()

    return loss



def loss_M(H,S_T,S_S,M=None):
    sample_loss_T = (H - S_T.detach()).pow(2)
    sample_loss_S = (H - S_S.detach()).pow(2)
    if M is not None:
        sample_loss = sample_loss_T*M +(1-M) *sample_loss_S
    else:
        sample_loss = (sample_loss_T + sample_loss_S)/2

    loss_data_all = sample_loss.detach()
    loss_data_T = sample_loss_T.detach()
    loss_data_S = sample_loss_S.detach()

    mask = torch.eye(loss_data_all .size(0), dtype=torch.bool, device=loss_data_all .device)  # 生成对角线为 True 的布尔矩阵
    loss_data_all [mask] = 0
    loss_data_T[mask] = 0
    loss_data_S[mask] = 0

    loss_data_all = loss_data_all.mean(dim=-1)
    loss_data_T = loss_data_T.mean(dim=-1)
    loss_data_S = loss_data_S.mean(dim=-1)

    loss=sample_loss.mean()

    return loss,loss_data_all.data ,loss_data_T.data,loss_data_S.data



