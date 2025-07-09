import time
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from evaluate import calc_map_k
from load_data import  load_dataset
from loss import ContrastiveLoss,loss_W,loss_M
from model import cal_similarityTag,ImgNet,TxtNet,High_sample
from utils import load_checkpoints, save_checkpoints, save_mat
from torch.optim import lr_scheduler
import datetime
class GCFH(object):
    def __init__(self, log,config):
        # self.M = None
        self.logger=log
        self.config=config
        self.dataset = config.dataset
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.nbits = config.hash_lens
        self._init_dataset()
        self._init_model()
        self.max_map = {'i2t': 0, "t2i": 0}

        self.global_step_losses = []  # 保存所有epoch的step loss
        self.global_step_counts = []  # 全局step计数（跨epoch）
        self.log_filename = self._generate_log_filename("loss")
        self.map_filename = self._generate_log_filename("map")

        data_num = len(self.train_loader.dataset)
        self.loss_T = torch.zeros(data_num).to(self.device)
        self.loss_S = torch.zeros(data_num).to(self.device)
        self.loss_all = torch.zeros(data_num).to(self.device)

    def _init_dataset(self):
        dataloader_train, dataloader_query, dataloader_retrival = load_dataset(self.config.dataset, self.config.data_pth,
                                                                   self.config.batch_size)
        self.train_loader = dataloader_train
        self.query_loader = dataloader_query
        self.retrieval_loader =  dataloader_retrival


    def _init_model(self):
        # hash layer
        self.ImageMlp = ImgNet(self.config.feat_lens, self.config.hash_lens).to(self.device)
        self.TextMlp = TxtNet(self.config.feat_lens, self.config.hash_lens).to(self.device)
        #params
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())

        total_param=(sum([param.nelement() for param in paramsImage])
                     +sum([param.nelement() for param in paramsText]))
        print('Total number of parameters: {}'.format(total_param))

        if self.dataset=="mscoco":
            self.optimizer_ImageMlp = optim.AdamW(paramsImage, lr=1e-3, betas=(0.5, 0.999))
            self.optimizer_TextMlp = optim.AdamW(paramsText, lr=1e-3, betas=(0.5, 0.999))
        else:

            self.optimizer_ImageMlp = optim.AdamW(paramsImage, lr=1e-3, betas=(0.5, 0.999))
            self.optimizer_TextMlp = optim.AdamW(paramsText, lr=1e-3, betas=(0.5, 0.999))
        #Define the mapping between dataset and scheduler parameters
        scheduler_params = {
            "flickr25k": {"milestones": [120, 320], "gamma": 1.2},
            "nus-wide": {"milestones": [30, 80], "gamma": 1.2},
            "mscoco": {"milestones": [200], "gamma": 0.6},
        }
        #Select parameters based on the dataset and create a scheduler
        params = scheduler_params.get(self.dataset)
        if params:
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, **params)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, **params)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        self.ContrastiveLoss = ContrastiveLoss(device=self.device)
        self.loss_l2 = torch.nn.MSELoss()
        self.S ,self.S_I_T,self.S_tag= cal_similarityTag(self.config,torch.as_tensor(self.train_loader.dataset.images, dtype=torch.float32),
                                                      torch.as_tensor(self.train_loader.dataset.texts, dtype=torch.float32),
                                torch.as_tensor(self.train_loader.dataset.tags, dtype=torch.float32),self.config.k_num,self.config.scale)

    def log_map_results(self, epoch, mapi2t, mapt2i):

        save_dir= "logs/map/"
        save_path = os.path.join(save_dir, self.map_filename)
        # 写入到 raw 文件（追加模式）
        with open(save_path , "a") as f:
            f.write(f"{epoch},{mapi2t.item()},{mapt2i.item()}\n")

    def _generate_log_filename(self,file_name):
        """生成带时间戳的日志文件名"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name=file_name+"_"+str(self.config.dataset)+"_"+str(self.config.hash_lens)+"bit_"+str(self.config.batch_size)+"bs_"+current_time+".raw"
        return file_name
    def save_loss_realtime(self, save_dir="logs/loss/"):
        """记录当前step的loss到日志文件"""
        # 确保日志目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, self.log_filename)

        # 如果是第一次写入，先写入表头
        if len(self.global_step_losses) == 1:
            with open(save_path, 'w') as f:
                f.write("step,loss\n")

        # 追加写入当前loss
        with open(save_path, 'a') as f:
            f.write(f"{len(self.global_step_losses)},{self.global_step_losses[-1]}\n")
    def train_epoch(self,epoch):
        self.ImageMlp.train()
        self.TextMlp.train()

        self.M = None



        if epoch>=self.config.eps_2 and epoch%1==0:

            sorted_values, sorted_indices = torch.sort(self.loss_all)
            # select_num = 3500
            select_num = 2500 * (1 + (epoch / 29))
            if int(select_num)>5000:
                select_num = 5000

            print("select_num", int(select_num))
            sort_ids = sorted_indices[int(select_num):]

            # 初始化为全False
            mask = torch.zeros_like(self.loss_all, dtype=torch.bool)
            # 将前面的大损失位置标记为True
            mask[sort_ids ] = True
            mask = mask.reshape(self.loss_all.shape)  # 恢复原始形状 (5000, 5000)
            #归一化

            self.loss_S = (self.loss_S - self.loss_S.min()) / (self.loss_S.max() - self.loss_S.min())
            self.loss_T=(self.loss_T-self.loss_T.min())/(self.loss_T.max() - self.loss_T.min())

            scale = 1
            scaled_diff=self.loss_S- self.loss_T
            scaled_diff = torch.sigmoid(scale *scaled_diff)

            self.M =torch.where(mask, scaled_diff,0.5)

            print(f"缩放后范围: [{scaled_diff.min().item():.4f}, {scaled_diff.max().item():.4f}]")
        running_loss = 0.0


        for idx, (img, txt, labl,index, tags) in enumerate(self.train_loader):
            img, txt ,tags= img.to(self.device), txt.to(self.device),tags.to(self.device)

            img_code = self.ImageMlp(img)
            text_code = self.TextMlp(txt)

            S = self.S[index, :][:, index].cuda()
            S_I_T=self.S_I_T[index, :][:, index].cuda()
            S_tag=self.S_tag[index, :][:, index].cuda()
            M = None
            if self.M is not None:
                M=self.M[index].cuda()
                row_M = M.unsqueeze(0).expand(M.size(0), -1)  # 插入行维度并扩展
                col_M = M.unsqueeze(1).expand(-1, M.size(0))  # 插入列维度并扩展
                M=(row_M +col_M)/2
            W =None
            if epoch>=self.config.eps_1:
                W = High_sample(S_tag,S_I_T,self.config.k)


            loss ,L_mse_spl_all,L_mse_spl_T ,L_mse_spl_S,printLoss= self.train_loss(img_code, text_code, S, S_I_T ,S_tag,W,M)
            for i in range(L_mse_spl_all.size(0)):
                self.loss_T[index[i]] = L_mse_spl_T[i]
                self.loss_S[index[i]]= L_mse_spl_S[i]
                self.loss_all[index[i]] =L_mse_spl_all[i]

            # 保存当前step的loss
            self.global_step_losses.append(printLoss.item())
            self.global_step_counts.append(len(self.global_step_losses))
            self.save_loss_realtime()

            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()

        return running_loss




    def train_loss(self,img_embedding,text_embedding,S_fus,S_I_T,S_tag,W,M):

        loss_cra = self.ContrastiveLoss(img_embedding, text_embedding )

        F_I = F.normalize(img_embedding, dim=1)
        F_T = F.normalize(text_embedding, dim=1)

        BI_BI = F_I.mm(F_I.t())
        BT_BT = F_T.mm(F_T.t())
        BI_BT = F_I.mm(F_T.t())
        BT_BI = F_T.mm(F_I.t())

        L_mse_fus = loss_W(BI_BI, S_fus, W) + loss_W(BT_BT, S_fus, W) + loss_W(BI_BT, S_fus, W) + loss_W(BT_BI, S_fus, W)

        loss_II, loss_II_all, loss_II_T, loss_II_S = loss_M(BI_BI, S_T=S_I_T, S_S=S_tag, M=M)
        loss_TT, loss_TT_all, loss_TT_T, loss_TT_S = loss_M(BT_BT, S_T=S_I_T, S_S=S_tag, M=M)
        loss_IT, loss_IT_all, loss_IT_T, loss_IT_S = loss_M(BI_BT, S_T=S_I_T, S_S=S_tag, M=M)
        loss_TI, loss_TI_all, loss_TI_T, loss_TI_S = loss_M(BT_BI, S_T=S_I_T, S_S=S_tag, M=M)


        L_mse_spl=loss_II+loss_TT+loss_IT+loss_TI

        L_mse_spl_all = (loss_II_all + loss_TT_all)+ (loss_IT_all+loss_TI_all)
        # _mse_spl_all = loss_II_all + loss_TT_all

        L_mse_spl_T = loss_II_T + loss_TT_T + loss_IT_T + loss_TI_T
        L_mse_spl_S = loss_II_S+ loss_TT_S + loss_IT_S  + loss_TI_S





        B = torch.sign((img_embedding + text_embedding))
        L_quant = (F.mse_loss(img_embedding, B) / img_embedding.shape[0] / self.config.hash_lens) + (F.mse_loss(text_embedding, B) / text_embedding.shape[0] / self.config.hash_lens)

        L_mse = L_mse_fus + L_mse_spl*self.config.beta
        loss = loss_cra +L_mse + L_quant
        printLoss = loss.detach()
        return loss,L_mse_spl_all,L_mse_spl_T,L_mse_spl_S,printLoss



    def eval_retrieval(self,epoch):

        test_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
        retrieval_dl_dict = {'img_code': [], 'txt_code': [], 'label': []}
        self.ImageMlp.eval()
        self.TextMlp.eval()
        with torch.no_grad():

            for _, (data_I, data_T, data_L, index) in enumerate(self.query_loader):

                data_I, data_T = data_I.cuda(), data_T.cuda()
                label_t = data_L.cuda()

                img_query = self.ImageMlp(data_I)
                txt_query = self.TextMlp(data_T)

                Im_code = torch.sign(img_query)
                Txt_code = torch.sign(txt_query)

                test_dl_dict['img_code'].append(Im_code)
                test_dl_dict['txt_code'].append(Txt_code)
                test_dl_dict['label'].append(label_t)

            for _, (data_I, data_T, data_L,index) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.cuda(), data_T.cuda()
                label_t_db= data_L.cuda()

                img_retrieval = self.ImageMlp(data_I)
                txt_retrieval = self.TextMlp(data_T)

                Im_code = torch.sign(img_retrieval)
                Txt_code = torch.sign(txt_retrieval)

                retrieval_dl_dict['img_code'].append(Im_code)
                retrieval_dl_dict['txt_code'].append(Txt_code)
                retrieval_dl_dict['label'].append(label_t_db)

        query_img = torch.cat(test_dl_dict['img_code'], dim=0).cpu()
        query_txt = torch.cat(test_dl_dict['txt_code'], dim=0).cpu()
        query_label = torch.cat(test_dl_dict['label'], dim=0).cpu()

        retrieval_img = torch.cat(retrieval_dl_dict['img_code'], dim=0).cpu()
        retrieval_txt = torch.cat(retrieval_dl_dict['txt_code'], dim=0).cpu()
        retrieval_label = torch.cat(retrieval_dl_dict['label'], dim=0).cpu()
     
        mapi2t = calc_map_k(query_img.cuda(), retrieval_txt.cuda(), query_label.cuda(), retrieval_label.cuda())
        mapt2i = calc_map_k(query_txt.cuda(), retrieval_img.cuda(), query_label.cuda(), retrieval_label.cuda())

        if mapi2t + mapt2i > self.max_map['i2t'] + self.max_map['t2i']:
            self.max_map['i2t'] = mapi2t
            self.max_map['t2i'] = mapt2i
            self.epoch_best = epoch
            save_checkpoints(self)
            save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, query_label, retrieval_label)
        self.logger.info("best epoch : {}".format(self.epoch_best))
        self.logger.info("max_mAPi2t : {}".format(self.max_map['i2t']))
        self.logger.info("max_mAPt2i : {}".format(self.max_map['t2i']))
        self.log_map_results(epoch, mapi2t, mapt2i)
        return mapi2t.item(), mapt2i.item()

    def train(self):
            self.max_map['i2t'] = self.max_map['t2i']
            I2T_MAP = []
            T2I_MAP = []


            for epoch in range(self.config.epoch):

                self.logger.info("=============== epoch: {}===============".format(epoch + 1))
                starimt = time.time()
                train_loss = self.train_epoch(epoch)

                end= time.time()
                print("epoch time: {}".format(end - starimt))
                self.logger.info("Training loss: {}".format(train_loss))
                if ((epoch + 1) % self.config.freq == 0) and (epoch + 1) > self.config.evl_epoch  :
                    self.logger.info("Testing...")
                    start=time.time()
                    img2text, text2img = self.eval_retrieval(epoch)
                    end = time.time()
                    print("检索花费时间", end - start)


                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))

    def test(self):
        load_checkpoints(self)
        img2text, text2img = self.eval_retrieval()
        self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))
