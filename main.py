import torch
import numpy as np
import argparse
from CFGH import GCFH
from utils import logger

if __name__ == '__main__':
    seeds = 2023
    torch.manual_seed(seeds) 
    torch.cuda.manual_seed(seeds)  
    torch.cuda.manual_seed_all(seeds)  
    np.random.seed(seeds)


    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset', type=str, default='nus-wide', help='Dataset name: mscoco, flickr25k, nus-wide')
    parser.add_argument('--data_pth', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--feat_lens',type=int, default=512)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--hash_lens', type=int, default=64
                        )
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--model_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--freq', type=int, default=1,help='eval interval')
    parser.add_argument('--evl_epoch', type=int, default=1,help='Number of epochs to start eval')
    parser.add_argument('--k_num', type=int, default=3500)  #flickr 3300 nus 3500  COCO 1000
    parser.add_argument('--scale', type=int, default=4000)  #4000

    parser.add_argument('--lamda1', type=int, default=0.8)
    parser.add_argument('--α1', type=float, default=0.35)
    parser.add_argument('--α2', type=float, default=1.5)#1.5
    parser.add_argument('--lamda2', type=float, default=0.8)# nus b16 0.8

    parser.add_argument('--gamma', type=float, default=0.5) #0.5
    parser.add_argument('--epsilon', type=float, default=0.08)#0.08

    parser.add_argument('--L1', type=float, default=1) #
    parser.add_argument('--L2', type=float, default=1)#

    config = parser.parse_args()


    log = logger(config)
    log.info('--- config: {}'.format(config))

    task = str(config.hash_lens) + " bits"
    log.info('=============== {}--{}--Total epochs:{} ==============='.format(config.dataset, task, config.epoch))
    model = GCFH(log ,config)
    if True:
        log.info('...Training is beginning...')
        model.train()
    else:
        log.info('...Test is beginning...')
        model.test()


