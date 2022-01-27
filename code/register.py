import world
import dataloader
import model
import utils
# from pathlib import Path
from pprint import pprint
from os.path import dirname, join
import os
from lastfm_dataloader import LastFM
from loader_dataloader import Loader
from datarate_dataloader import DataRate

config = world.config
world_config = world.world_config

if world_config['dataset'] in ['gowalla', 'yelp2018']:
    # path = Path(__file__).parent.parent / 'data' / world.dataset
    path = join(dirname(os.path.dirname(__file__)), 'data', world_config['dataset'])
    # 导入数据集
    dataset = Loader(path=path)
elif world_config['dataset'] == 'lastfm':
    #  path = Path(__file__).parent.parent / 'data' / 'lastfm'
    path = join(dirname(os.path.dirname(__file__)), 'data', 'lastfm')
    dataset = LastFM(path=path)
elif world_config['dataset'] in ['amazon-electronic', 'amazon-book', 'amazon-book-init', 'movielen']:
    path = join(dirname(os.path.dirname(__file__)), 'data', world_config['dataset'])
    dataset = DataRate(path=path)

# 这一步是显示config
print('===========config================')
# pprint(world.config)
print("cores for test:", world_config['CORES'])
print("comment:", world_config['comment'])
print("tensorboard:", world_config['tensorboard'])
print("LOAD:", world_config['LOAD'])
print("Weight path:", world_config['PATH'])
print("Test Topks:", world_config['topks'])
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'ngcf': model.NGCF,
    'neumf': model.NeuMF,
    'cmn': model.CMN,
    'cf_mo': model.CF_MO,
    'dhcf': model.DHCF
}

LOSSES = {
    'bpr': utils.BPRLoss,
    'score_loss': utils.ScoreLoss,
}
