import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from evaluate import calc_map_k
from load_data import  load_dataset
from loss import ContrastiveLoss,loss_w
from model import cal_similarityTag,ImgNet,TxtNet,High_sample
from utils import load_checkpoints, save_checkpoints, save_mat
from torch.optim import lr_scheduler

class GCFH(object):
    def __init__(self, log,config):
        self.logger=log
        self.config=config
        self.dataset = config.dataset
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.nbits = config.hash_lens
        self._init_dataset()
        self._init_model()
        self.max_map = {'i2t': 0, "t2i": 0}

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

        #optimizer  coco需要
        # self.optimizer_ImageMlp = optim.AdamW(paramsImage, lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)
        # self.optimizer_TextMlp = optim.AdamW(paramsText, lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)

        # #cocoTOnus 16 不需要weight_decay ？？？
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

        # self.e = 1

    def train_epoch(self):
        self.ImageMlp.train()
        self.TextMlp.train()

        # self.ImageMlp.set_alpha(self.e)
        # self.TextMlp.set_alpha(self.e)
        # self.e+=1
        running_loss = 0.0
        for idx, (img, txt, labl,index, tags) in enumerate(self.train_loader):
            img, txt ,tags= img.to(self.device), txt.to(self.device),tags.to(self.device)

            img_code = self.ImageMlp(img)
            text_code = self.TextMlp(txt)

            S = self.S[index, :][:, index].cuda()
            S_I_T=self.S_I_T[index, :][:, index].cuda()
            S_tag=self.S_tag[index, :][:, index].cuda()

            W = High_sample(S_tag,S_I_T,self.config)
            loss = self.train_loss(img_code, text_code, S, W)

            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()
            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()

        return running_loss











    def train_loss(self,img_embedding,text_embedding,S,W):

        loss_cra = self.ContrastiveLoss(img_embedding, text_embedding )

        F_I = F.normalize(img_embedding, dim=1)
        F_T = F.normalize(text_embedding, dim=1)

        BI_BI = F_I.mm(F_I.t())
        BT_BT = F_T.mm(F_T.t())
        BI_BT = F_I.mm(F_T.t())
        BT_BI = F_T.mm(F_I.t())

        loss_s = loss_w(BI_BI, S, W) + loss_w(BT_BT, S, W) + loss_w(BI_BT, S, W) + loss_w(BT_BI, S, W)

        loss_cons = self.loss_l2(BI_BT, BI_BI) + \
                    self.loss_l2(BI_BT, BT_BT) + \
                    self.loss_l2(BI_BI, BT_BT) + \
                    self.loss_l2(BI_BT, BI_BT.t())

        B = torch.sign((img_embedding + text_embedding))

        loss_quant = (F.mse_loss(img_embedding, B) / img_embedding.shape[0] / self.config.hash_lens) + (F.mse_loss(text_embedding, B) / text_embedding.shape[0] / self.config.hash_lens)

        loss = loss_cra  +  loss_s  + loss_cons + loss_quant
        return loss



    def eval_retrieval(self):

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
            save_checkpoints(self)
            save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, query_label, retrieval_label)
        self.logger.info("max_mAPi2t : {}".format(self.max_map['i2t']))
        self.logger.info("max_mAPt2i : {}".format(self.max_map['t2i']))
        return mapi2t.item(), mapt2i.item()

    def train(self):
            self.max_map['i2t'] = self.max_map['t2i']
            I2T_MAP = []
            T2I_MAP = []

            starimt=time.time()
            for epoch in range(self.config.epoch):
                self.logger.info("=============== epoch: {}===============".format(epoch + 1))
                train_loss = self.train_epoch()
                self.logger.info("Training loss: {}".format(train_loss))
                end= time.time()
                print("epoch time: {}".format(end - starimt))

                if ((epoch + 1) % self.config.freq == 0) and (epoch + 1) > self.config.evl_epoch  :
                    self.logger.info("Testing...")
                    img2text, text2img = self.eval_retrieval()
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))

    def test(self):
        load_checkpoints(self)
        img2text, text2img = self.eval_retrieval()
        self.logger.info('I2T: {}, T2I: {}'.format(img2text, text2img))
