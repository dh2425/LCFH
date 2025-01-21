import torch
import torch.optim as optim
import torch.nn.functional as F
from evaluate import calc_map_k
from load_data import  load_dataset
from loss import ContrastiveLoss
from model import cal_similarityTag,ImgNet,TxtNet
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
        dataloader_train ,dataloader_query_retrival= load_dataset(self.config.dataset, self.config.data_pth,self.config.batch_size)
        self.train_loader =  dataloader_train
        self.query_loader = dataloader_query_retrival['query']
        self.retrieval_loader = dataloader_query_retrival['retrieval']

    def _init_model(self):
        self.ImageMlp = ImgNet(self.config.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TxtNet(self.config.feat_lens, self.nbits).to(self.device)
        #params
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())

        #hash layer
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-3, betas=(0.5, 0.999))

        if self.dataset == "flickr25k" :
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[120, 320], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[120, 320], gamma=1.2)

        elif self.dataset == "nus-wide":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)

        elif self.dataset == "mscoco":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[200], gamma=0.6)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[200], gamma=0.6)

        self.ContrastiveLoss = ContrastiveLoss(device=self.device)
        self.loss_l2 = torch.nn.MSELoss()
        self.S = cal_similarityTag(self.config,torch.as_tensor(self.train_loader.dataset.images, dtype=torch.float32),
                                                      torch.as_tensor(self.train_loader.dataset.texts, dtype=torch.float32),
                                torch.as_tensor(self.train_loader.dataset.tags, dtype=torch.float32),self.config.k_num,self.config.α)


    def train_epoch(self):
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        for idx, (img, txt, labl,index, tags,len_tags) in enumerate(self.train_loader):
            img, txt ,tags= img.to(self.device), txt.to(self.device),tags.to(self.device)

            img_embedding = self.ImageMlp(img)
            text_embedding = self.TextMlp(txt)

            loss_cra = self.ContrastiveLoss(img_embedding, text_embedding)
            S = self.S[index, :][:, index].cuda()

            F_I = F.normalize(img_embedding, dim=1)
            F_T = F.normalize(text_embedding, dim=1)

            BI_BI = F_I.mm(F_I.t())
            BT_BT = F_T.mm(F_T.t())
            BI_BT = F_I.mm(F_T.t())
            BT_BI = F_T.mm(F_I.t())

            loss_s= self.loss_l2(BI_BI, S) + self.loss_l2(BT_BT, S)  + self.loss_l2(BI_BT, S) + self.loss_l2(BT_BI, S)

            loss_cons =self.loss_l2(BI_BT, BI_BI) + \
                        self.loss_l2(BI_BT, BT_BT) + \
                        self.loss_l2(BI_BI, BT_BT) + \
                        self.loss_l2(BI_BT, BI_BT.t())


            B = torch.sign((img_embedding + text_embedding))
            loss_quant =( F.mse_loss(img_embedding, B) / img_embedding.shape[0] / self.nbits) + (F.mse_loss(text_embedding,B) / text_embedding.shape[0] / self.nbits)



            loss = loss_cra  +  loss_s + loss_cons+ loss_quant
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()

            loss.backward()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()

            running_loss += loss.item()

            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()

        return running_loss


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
            for epoch in range(self.config.epoch):
                self.logger.info("=============== epoch: {}===============".format(epoch + 1))
                train_loss = self.train_epoch()
                self.logger.info("Training loss: {}".format(train_loss))
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
