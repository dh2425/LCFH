import logging
import os
import pickle
from os.path import join
import torch


def save_checkpoints(self):

    file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
    file_path = os.path.join(self.config.model_dir, file_name)
    os.makedirs(self.config.model_dir, exist_ok=True)

    obj = {
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict(),

        }
    torch.save(obj, file_path)
    self.logger.info('**Save the model successfully.**')




def load_checkpoints(self):

    file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
    file_path = os.path.join(self.config.model_dir, file_name)

    try:
        obj = torch.load(file_path, map_location= self.device)

        self.logger.info('**************** Load checkpoint %s ****************', file_path)
    except IOError:
        self.logger.error('********** Fail to load checkpoint %s! *********', file_path)

        raise IOError
    self.ImageMlp.load_state_dict(obj['ImageMlp'])
    self.TextMlp.load_state_dict(obj['TextMlp'])


def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt,query_labels, retrieval_labels):
    file_name = self.dataset + '_hash_' + str(self.nbits) + ".pkl"
    os.makedirs(self.config.result_dir, exist_ok=True)
    query_img = query_img.cpu().detach().numpy()
    query_txt = query_txt.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()
    retrieval_txt = retrieval_txt.cpu().detach().numpy()
    query_labels = query_labels.cpu().detach().numpy()
    retrieval_labels =retrieval_labels.cpu().detach().numpy()

    result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
    }

    with open(join(self.config.result_dir, file_name), 'wb') as f:
        pickle.dump(result_dict, f)




def logger(config):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = config.dataset + '_' + str(config.hash_lens)+ '.log'
    log_dir = './logs'
    txt_log = logging.FileHandler(os.path.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger


