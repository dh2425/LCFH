import os

from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs,tags,tags_len):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.tags = tags
        self.tags_len=tags_len
   
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        tags=self.tags[index]
        tags_len=self.tags_len[index]
        return img, text, lab, index,tags,tags_len

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(dataset,data_pth, batch_size):
    '''
        load datasets : flickr25k, mscoco, nus-wide
    '''

    # train_loc = r'E:\selfpapper3\data\pkl\TC10_25k4TOnuswide\train_TC10_25k4TOnuswide.pkl'
    # query_loc = r'E:\selfpapper3\data\pkl\TC10_25k4TOnuswide\query_TC10_25k4TOnuswide.pkl'
    # retrieval_loc = r'E:\selfpapper3\data\pkl\TC10_25k4TOnuswide\retrieval_TC10_25k4TOnuswide.pkl'

    file_path = os.path.join(data_pth, dataset)

    train_loc =file_path+r'\train.pkl'
    query_loc =file_path+r'\query.pkl'
    retrieval_loc =file_path+r'\retrieval.pkl'


    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['label'],dtype=torch.int64)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)
        train_images = torch.tensor(data['image'], dtype=torch.float32)
        train_tags = torch.tensor(data['tag'], dtype=torch.float32)
        train_tags_len = torch.tensor(data['tag_len'], dtype=torch.float32)
    
    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'],dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)    
        query_images = torch.tensor(data['image'], dtype=torch.float32)
        query_tags = torch.tensor(data['tag'], dtype=torch.float32)
        query_tags_len = torch.tensor(data['tag_len'], dtype=torch.float32)
      
    with open(retrieval_loc,'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_lables = torch.tensor(data['label'],dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32) 
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)
        retrieval_tags = torch.tensor(data['tag'], dtype=torch.float32)
        retrieval_tags_len = torch.tensor(data['tag_len'], dtype=torch.float32)

    imgs = {'train': train_images, 'query': query_images, 'retrieval': retrieval_images}
    texts = {'train': train_texts,  'query': query_texts, 'retrieval': retrieval_texts}
    labs = {'train': train_labels, 'query': query_labels, 'retrieval': retrieval_lables}
    tags= {'train': train_tags, 'query': query_tags, 'retrieval':  retrieval_tags}
    tags_len={'train': train_tags_len, 'query': query_tags_len, 'retrieval':  retrieval_tags_len}


    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x],tags=tags[x],tags_len=tags_len[x]) for x in ['train', 'query', 'retrieval']}
    shuffle = {'train': True, 'query': False, 'retrieval': False}
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, drop_last=True, pin_memory=True,shuffle=shuffle[x], num_workers=4) for x in ['train', 'query', 'retrieval']}
    return dataloader

