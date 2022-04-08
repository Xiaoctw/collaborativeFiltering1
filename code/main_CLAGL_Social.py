import world
import utils
from world import cprint
import torch
import numpy as np
import sys
import time
import Procedure
from os.path import join
import model
import os
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
# args = config.parse_args()
world_config = world.world_config
config = world.config
# print(world_config)
# print(config)
if not os.path.exists(world_config['FILE_PATH']):
    os.makedirs(world_config['FILE_PATH'], exist_ok=True)
# ==============================
utils.set_seed(world_config['seed'])
print(">>SEED:", world_config['seed'])
# ==============================


world_config['model_name']='clagl_social'
world_config['comment'] = world_config['model_name']


from register import dataset

print('---recModel---')
recModel = model.CLAGL_Social(config, dataset)
print('--recModel finished---')

recModel = recModel.to(world_config['device'])
# print('device:{}'.format(world_config['device']))

# print(recModel.parameters())
loss = utils.ScoreLoss(recModel, config)
weight_file = utils.getFileName()
print('load and save to {}'.format(weight_file))

if world_config['LOAD']:
    try:
        # 导入现有模型
        recModel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world_config['tensorboard']:
    summary_path = world_config['BOARD_PATH'] + '/' + (
            time.strftime("%m-%d-%Hh%Mm%Ss-") + world_config['dataset'] + '-' + world_config['comment'])
    from tensorboardX import SummaryWriter

    w: SummaryWriter = SummaryWriter(str(summary_path))
# join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    cprint("not enable tensorflowboard")

try:
    # recModel.train_mul()
    recall_list = []
    ndcg_list = []
    for epoch in range(world_config['TRAIN_epochs']):
        start = time.time()
        if epoch % 10 == 0:
            cprint('[TEST]')
            result = Procedure.Test(dataset, recModel, epoch, w, config['multicore'])
            recall_list.append(float("{:.3f}".format(float(result['recall'][0]))))
            ndcg_list.append(float("{:.3f}".format(float(result['ndcg'][0]))))
        # recModel.train_attn_weight()
        # prnt('begin train')
        Procedure.Score_train_original(dataset, recModel, loss, epoch, Neg_k, w)
    result= Procedure.Test(dataset, recModel, world_config['TRAIN_epochs'], w, config['multicore'])
    recall_list.append(float("{:.3f}".format(float(result['recall'][0]))))
    ndcg_list.append(float("{:.3f}".format(float(result['ndcg'][0]))))
    print('recall:{}'.format(recall_list))
    print('ndcg:{}'.format(ndcg_list))
    torch.save(recModel.state_dict(), weight_file)
finally:
    if world_config['tensorboard']:
        w.close()
