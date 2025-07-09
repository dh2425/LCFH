import json
import os
import numpy as np
from os.path import join


img_path = 'all/all_imgs.txt'
txt_path = 'all/all_tags.txt'
lab_path = 'all/all_labels.txt'

with open(img_path, 'r', encoding='utf-8') as f:
    img_datas =np.array([line.strip() for line in f])
    # img_datas = f.readlines()

with open(txt_path, 'r', encoding='utf-8') as f:
    txt_datas  = np.array([line.strip() for line in f])
    # txt_datas  = f.readlines()

lab_datas  = np.loadtxt(lab_path, dtype=np.int32)

first = True
for label in range(lab_datas.shape[1]):
    #属于第label个类别样本的下标
    index = np.where(lab_datas[:, label] == 1)[0]
    N = index.shape[0]
    perm = np.random.permutation(N) #生成了一个从0到N-1的随机排列数组
    index = index[perm]

    if first:
        query_index = index[:160]
        train_index = index[160:160 + 400]
        first = False
    else:
        ind = np.array([i for i in list(index) if i not in (list(train_index) + list(query_index))]) #检查i是否不在train_index和query_index的合并列表中。 目的是不要出现重复样本
        query_index = np.concatenate((query_index, ind[:80]))
        train_index = np.concatenate((train_index, ind[80:80 + 200]))


database_index = np.array([i for i in list(range(lab_datas.shape[0])) if i not in list(query_index)])
if train_index.shape[0] < 5000:
    pick = np.array([i for i in list(database_index) if i not in list(train_index)])
    N = pick.shape[0]
    perm = np.random.permutation(N)
    pick = pick[perm]
    res = 5000 - train_index.shape[0]
    train_index = np.concatenate((train_index, pick[:res]))




# 创建保存数据的函数
def save_dataset(data_type, img_data, txt_data, label_data):
    # 创建目录
    os.makedirs(data_type, exist_ok=True)

    # 保存图片路径
    with open(join(data_type, 'images.raw'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(img_data))

    # 保存文本数据
    with open(join(data_type, 'texts.raw'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_data))

    # 保存标签数据
    with open(join(data_type, 'labels.raw'), 'w', encoding='utf-8') as f:
        for label in label_data:
            f.write(' '.join(map(str, label)))  # 将numpy数组转为字符串
            f.write('\n')

# 获取当前工作目录
P = os.getcwd()
print(f"当前工作目录: {P}")

# 保存数据集
save_dataset(join(P, 'train'),
             img_datas[train_index],
             txt_datas[train_index],
             lab_datas[train_index])

save_dataset(join(P, 'query'),
             img_datas[query_index],
             txt_datas[query_index],
             lab_datas[query_index])

save_dataset(join(P, 'retrieval'),
             img_datas[database_index],
             txt_datas[database_index],
             lab_datas[database_index])

print("数据已成功保存到对应文件夹！")