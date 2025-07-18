import json
import random
import time

import numpy as np
import torch
from os.path import join
from PIL import Image
import clip
from tqdm import tqdm
import os
import json
from collections import Counter

start=time.time()

def seed_torch(seed=27):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()




device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


train_img_path = '../raw/train/images.txt'
train_lab_path = '../raw/train/labels.txt'
train_txt_path = '../raw/train/texts.txt'
# train_generate_tags_path= "../from_generate_makePkl/generated_captions_epoch_4.source_labels"
train_generate_tags_path= "../../../Fine tuning/generated_Snuswide_Tflickr25k.json"

query_img_path = '../raw/query/images.txt'
query_lab_path = '../raw/query/labels.txt'
query_txt_path = '../raw/query/texts.txt'

retrival_img_path = '../raw/retrieval/images.txt'
retrival_lab_path = '../raw/retrieval/labels.txt'
retrival_txt_path = '../raw/retrieval/texts.txt'




# TODO 修改为包含所有图片(约25K)的文件夹的绝对路径
img_root_path = r"E:\dataset\hashing\MITH\nuswide\images"










def Pseudo_label_one_hot(generate_label_path):

    with open(generate_label_path, 'r', encoding='utf-8') as file:
        # 使用json.load()解析文件内容
        data = json.load(file)

     # 提取所有单词并创建词表
    word_counts = Counter()
    for item in data:
        words = item['caption_g'].split(', ')
        word_counts.update(words)

    # 获取前k个频率最高的单词
    k = 24
    # 最小词频
    min_freq=50
    vocab = [word for word, count in word_counts.most_common(k) if count >= min_freq]
    # 制作one-hot编码
    one_hot_encodings = []
    for item in data:
        words = item['caption_g'].split(', ')
        encoding = [1 if word in words else 0 for word in vocab]
        one_hot_encodings.append(encoding)

    return np.array(one_hot_encodings)








def caption2str(caption_filename):
    """
    # return ndarray : (N, ), each one is a str of tags, i.e., 'cigarette tattoos smoke red dress sunglasses'
    """
    with open(caption_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    N_TXT=len(lines)
    all_text_features=np.zeros((N_TXT,512))
    with torch.no_grad():
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "":
                print("Error: Empty tags...")  # no empty in flickr
                exit()

            tags = line.split(',')
            sss = " ".join(tags)
            #将文本转为token
            text = clip.tokenize(sss.strip()[:77]).to(device)
            text_features = model.encode_text(text).cpu().numpy()#（1,512）
            all_text_features[i]=text_features
            freq=1000
            if i%freq==0:
                print(f"{i}/{len(lines)}")
    return  all_text_features,lines




def images_clip(img_path):
    img_abs_paths = []  # python list (N), each one is str 'mirflickr/im1.jpg'
    with open(img_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            ttt = line.strip()
            img_abs_paths.append(os.path.join(img_root_path, ttt.split("/")[1]))  # abs path.

    img_abs_paths = np.array(img_abs_paths)  # ndarray: (N,)
    N_IMG = len(img_abs_paths)
    print("#images:", N_IMG)
    all_img_features = np.zeros((N_IMG, 512))
    with torch.no_grad():
        for i in tqdm(range(N_IMG)):
            im = preprocess(Image.open(img_abs_paths[i])).unsqueeze(0).to(device)
            all_img_features[i] = model.encode_image(im).cpu().numpy()
    return all_img_features



def mak_data(img_path,txt_path,lab_path,lab_g_path=None):

    all_txt_features,caption_orl = caption2str(txt_path)  # ndarray: (N,)
    print("---------------------------raw-----------------------------------------")
    print("all_txt_features: ", all_txt_features.shape)


    all_lab = np.loadtxt(lab_path, dtype=np.float32)  # ndarray: (N, 24)
    print("-----------------------------labels---------------------------------------")
    print("all_lab: ", all_lab.shape)
    print("0th all_lab: ", all_lab[0])

    print("--------------------------images------------------------------------------")
    all_img_features = images_clip(img_path)
    print("all_img_features: ",all_img_features.shape)

    if lab_g_path!=None:
        all_lab_g =Pseudo_label_one_hot(lab_g_path)
        print("-----------------------------labels---------------------------------------")
        print("lab_g: ", all_lab_g .shape)
        print("lab_g: ",all_lab_g [0])

        return all_img_features, all_txt_features,all_lab,all_lab_g
    else:
        return all_img_features, all_txt_features,all_lab


import pickle
P = os.getcwd()
print(P)
print("train")
train_img, train_txt, train_lab, train_lab_g = mak_data(train_img_path, train_txt_path, train_lab_path,train_generate_tags_path)
train_data = {'image': train_img, 'text': train_txt,'label':  train_lab, 'label_g':train_lab_g}
with open(join(P, "train_flickrTOnuswide.pkl"), 'wb') as f:
    pickle.dump(train_data, f)
end=time.time()
print("特征提取花费时间：",end-start)

# print("query")
# query_img, query_txt, query_lab = mak_data(query_img_path,query_txt_path,query_lab_path)
# query_data = {'image': query_img, 'text': query_txt,'label': query_lab}
# with open(join(P, "query_flickrTOnuswide.pkl"), 'wb') as f:
#     pickle.dump(query_data, f)
#
# print("retrival")
# retrival_img, retrival_txt, retrival_lab = mak_data(retrival_img_path ,retrival_txt_path,retrival_lab_path)
# retrival_data={'image': retrival_img, 'text': retrival_txt,'label': retrival_lab}
# with open(join(P, "retrival_flickrTOnuswide.pkl"), 'wb') as f:
#     pickle.dump(retrival_data, f)


'''
拆分数据集为[query, train, retrieval]
'''








