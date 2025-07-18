import os
from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch

class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs,tags=None):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.tags = tags

   
    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        if self.tags is not None :
            tags=self.tags[index]
            return img, text, lab, index, tags
        else:
            return img, text, lab, index

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(dataset,data_pth, batch_size):
    '''
        load datasets : flickr25k, mscoco, nus-wide
    '''

    file_path = os.path.join(data_pth, dataset)

    train_loc =file_path+r'\train.pkl'
    query_loc =file_path+r'\query.pkl'
    retrieval_loc =file_path+r'\retrieval.pkl'







    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = torch.tensor(data['label'],dtype=torch.int64)
        train_texts = torch.tensor(data['text'], dtype=torch.float32)
        train_images = torch.tensor(data['image'], dtype=torch.float32)
        train_tags = torch.tensor(data['label_g'], dtype=torch.int64)

    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'], dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)
        query_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_lables = torch.tensor(data['label'], dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32)
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)







    dataset_train =CustomDataSet(images=train_images, texts=train_texts, labs=train_labels, tags=train_tags)
    dataset_query = CustomDataSet(images=query_images, texts=query_texts, labs=query_labels)
    dataset_retrival = CustomDataSet(images=retrieval_images, texts=retrieval_texts, labs=retrieval_lables)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, drop_last=True, pin_memory=True,shuffle=True)
    dataloader_query = DataLoader(dataset_query, batch_size=batch_size,  drop_last=True, pin_memory=True, shuffle=False)
    dataloader_retrival = DataLoader(dataset_retrival, batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=False)


    return dataloader_train ,dataloader_query,dataloader_retrival

