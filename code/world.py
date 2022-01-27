import os
from os.path import join, dirname
import torch
from enum import Enum
import sys
import multiprocessing
from pathlib import Path
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Go collaborative filter models")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--l2', type=float, default=1e-3,  # L2正则化系数
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.8,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='lastfm',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book,amazon-electronic,amazon-book-init,movielen]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20,40,60]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    #  parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn, ngcf,neumf,cmn,cf_mo]')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--w1', type=float, default=1.)
    parser.add_argument('--w2', type=float, default=0.1)
    parser.add_argument('--attn_weight', type=int, default=0)
    parser.add_argument('--comment', type=str, default='None')
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--leaky_alpha', type=float, default=0.2)
    parser.add_argument('--reg_alpha', type=float, default=0.1)
    parser.add_argument('--loss_mode', type=str, default='mse',
                        help="cf_mo loss mode, default is mse, we can choose bce")
    parser.add_argument('--neighbor_num', type=int, default=8, help='Number of neighbor nodes')
    # 这些是对一阶关系图进行筛选时，采用的部分参数。
    parser.add_argument('--num_path', default=32, help='number of path per node')
    parser.add_argument('--path_length', default=32, help='length of path')
    parser.add_argument('--restart_alpha', type=float, default=0.5)
    parser.add_argument('--adj_top_k', type=int, default=16)
    parser.add_argument('--multi_action', type=int, default=0)
    parser.add_argument('--distance_measure', type=str, default='occurrence', choices=['occurrence',
                                                                                       'inverted',
                                                                                       'cosine_similarity'])

    parser.add_argument('--sample_neg', type=int, default=0, help='Whether to select to sample'
                                                                  ' samples according to frequency of occurrence')

    parser.add_argument('--delete_user', type=int, default=0, help='Whether to delete users to test cold start')

    # DGCF的部分参数
    parser.add_argument('--factors', type=int, default=8)
    parser.add_argument('--pick', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=2)  # 这么少的吗？

    # SMP，也就是加入非线性的部分参数
    parser.add_argument('--ori_temp', type=float, default=0.7)
    parser.add_argument('--min_temp', type=float, default=0.01)
    parser.add_argument('--gum_temp_decay', type=float, default=0.005)
    parser.add_argument('--epoch_temp_decay', type=int, default=1, )
    parser.add_argument('--division_noise', type=float, default=3, )

    # SSL参数
    parser.add_argument('--ssl_ratio', type=float, default=0.5)
    parser.add_argument('--ssl_temp', type=float, default=0.5)
    parser.add_argument('--ssl_reg', type=float, default=0.5)
    parser.add_argument('--ssl_mode', type=str, default='both_side')
    return parser.parse_args()


args = parse_args()

print('delete_user:{}'.format(args.delete_user))

"""
config
"""
config = {'batch_size': args.bpr_batch, 'latent_dim_rec': args.recdim, 'n_layers': args.layer,
          'dropout': args.dropout,
          'keep_prob': args.keepprob, 'A_n_fold': args.a_fold,
          'test_u_batch_size': args.testbatch, 'multicore': args.multicore,
          'lr': args.lr, 'decay': args.decay,
          'pretrain': args.pretrain, 'A_split': False, 'bigdata': False,
          'num_experts': args.num_experts, 'leaky_alpha': args.leaky_alpha,
          'attn_weight': args.attn_weight,
          'reg_alpha': args.reg_alpha,
          'w1': args.w1,
          'w2': args.w2,
          'neighbor_num': args.neighbor_num,
          'loss_mode': args.loss_mode,
          'num_path': args.num_path,
          'path_length': args.path_length,
          'restart_alpha': args.restart_alpha,
          'adj_top_k': args.adj_top_k,
          'distance_measure': args.distance_measure,
          'multi_action': args.multi_action,
          'factors': args.factors,
          'pick': args.pick,
          'n_iterations': args.n_iterations,
          'l2_normalize': args.l2,
          'ori_temp': args.ori_temp,
          'min_temp': args.min_temp,
          'gum_temp_decay': args.gum_temp_decay,
          'epoch_temp_decay': args.epoch_temp_decay,
          'division_noise': args.division_noise,
          'sample_neg': args.sample_neg,
          'delete_user': args.delete_user,
          'ssl_ratio': args.ssl_ratio,
          'ssl_temp': args.ssl_temp,
          'ssl_reg': args.ssl_reg,
          'ssl_mode':args.ssl_mode
          }
# config['batch_size'] = 4096


world_config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'amazon-electronic', 'amazon-book-init', 'movielen']
all_models = ['mf', 'lgn', 'ngcf', 'neumf', 'cmn', 'cf_mo', 'dhcf']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# args = parse_args()
world_config['args'] = args
ROOT_PATH = dirname(dirname(__file__))
CODE_PATH = ROOT_PATH + '/code'
DATA_PATH = ROOT_PATH + '/data'
BOARD_PATH = CODE_PATH + '/runs'
FILE_PATH = CODE_PATH + '/checkpoints'
world_config['ROOT_PATH'] = ROOT_PATH
world_config['CODE_PATH'] = CODE_PATH
world_config['DATA_PATH'] = DATA_PATH
world_config['BOARD_PATH'] = BOARD_PATH
world_config['FILE_PATH'] = FILE_PATH
GPU = torch.cuda.is_available()
# print('GPU is available:{}'.format(GPU))
world_config['GPU'] = GPU
device = torch.device('cuda:{}'.format(args.cuda) if GPU and args.cuda else "cpu")
# device=torch.device('cpu')
world_config['device'] = device
CORES = multiprocessing.cpu_count() // 2
world_config['CORES'] = CORES
seed = args.seed
world_config['seed'] = seed
dataset = args.dataset
world_config['dataset'] = dataset
model_name = args.model
world_config['model_name'] = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")
TRAIN_epochs = args.epoch
world_config['TRAIN_epochs'] = TRAIN_epochs
LOAD = args.load
world_config['LOAD'] = LOAD
PATH = args.path
world_config["PATH"] = PATH
topks = eval(args.topks)
world_config['topks'] = topks
tensorboard = args.tensorboard
world_config['tensorboard'] = tensorboard
# comment = args.comment
if args.comment == 'None':
    world_config['comment'] = model_name
else:
    world_config['comment'] = args.comment
if world_config['model_name'] in {'mf', 'lgn', 'ngcf', 'neumf', 'cmn', "dhcf"}:
    world_config['loss'] = 'bpr'
elif world_config['model_name'] == 'dgcf':
    world_config['loss'] = 'dgcf_loss'
else:  # cf_mo, bpr_cfig
    world_config['loss'] = 'score_loss'

world_config['train_hard'] = False
world_config['test_hard'] = True
test_hard = True


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")
