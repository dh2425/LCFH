import json
import random
import numpy as np
import torch
import os
from os.path import join
from os import listdir
from PIL import Image

from tqdm import tqdm
import re
from collections import Counter


seed = 1
torch.random.manual_seed(seed=seed)
np.random.seed(seed=seed)
random.seed(seed)


img_path = '../raw/train/images.txt'
txt_path = '../raw/train/texts.txt'
lab_path = '../raw/train/labels.txt'



"""图片路径"""
with open(img_path, 'r', encoding='utf-8') as f:
    img_path_lines = f.readlines()



CLASSES_21 = [
'animal','beach','buildings','clouds','flowers','grass','lake','mountain','ocean','person',
'plants','reflection','road','rocks','sky','snow','sunset','tree','vehicle','water','window'
]


CLASSES = ['sky','clouds','person',  'water',  'animal','grass','buildings', 'window' ,'plants' , 'lake ']
labels = np.loadtxt(lab_path, dtype=np.float32)

all_set = []
for i,label in enumerate(labels):
    sample_dict={}
    categorys = []
    for index,label_value in enumerate(label):
        if label_value==1:
            categorys.append(CLASSES[index])
    # tags
    tags = ", ".join(categorys)
    sample_dict["caption"] =tags
    all_set.append(sample_dict)




 # 提取所有单词并创建词表
word_counts = Counter()
for item in all_set:
    words = item['caption'].split(', ')
    word_counts.update(words)
print(word_counts)



after_labels=labels

after_img_path=img_path_lines





all_set = []
for i,label in enumerate(after_labels):
    sample_dict={}
    categorys = []
    for index,label_value in enumerate(label):
        if label_value==1:
            categorys.append(CLASSES[index])

    tags = ", ".join(categorys)
    sample_dict["caption"] =tags


    img_path=after_img_path[i].rstrip('\n')

    img_path = img_path.split('/')[1]


    sample_dict["image"] =img_path

    filename_without_extension = img_path .split('.')[0]

    image_index= filename_without_extension.split('m')[-1]

    image_index=int(image_index)


    sample_dict["image_id"] = image_index

    all_set.append(sample_dict)

# 将JSON字符串写入到文件中
with open('nuswide_train_labels.json', 'w') as json_file:
    json.dump(all_set, json_file)

