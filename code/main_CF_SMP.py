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

world_config['comment'] = 'cf_smp'
world_config['model_name'] = 'cf_smp'

from register import dataset

print('---recModel---')
recModel = model.CF_SMP(config, dataset)
print('--recModel finished---')

recModel = recModel.to(world_config['device'])
loss = utils.BPRLoss(recModel, config)

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
import math

gum_temp = config['ori_temp']
# print(recModel)
try:
    for epoch in range(world_config['TRAIN_epochs']):
        start = time.time()
        if epoch % config['epoch_temp_decay'] == 0:
            # Temp decay
            gum_temp = config['ori_temp'] * math.exp(-config['gum_temp_decay'] * epoch)
            gum_temp = max(gum_temp, config['min_temp'])
            # print('decay gum_temp:', gum_temp)
        if epoch % 10 == 0:
            cprint('[TEST]')
            Procedure.Test(dataset, recModel, epoch, w, config['multicore'], gum_temp, world_config['test_hard'])
        # recModel.train_attn_weight()
        Procedure.SMP_train_original(dataset=dataset, recommend_model=recModel, loss_class=loss, epoch=epoch, w=w,
                                     gum_temp=gum_temp,
                                     hard=world_config['train_hard'],)
    Procedure.Test(dataset, recModel, world_config['TRAIN_epochs'], w, config['multicore'], gum_temp,
                   world_config['test_hard'])

    # world_config['TRAIN_epochs']=2*world_config['TRAIN_epochs']
    # recModel.train_MMoE()
    # for epoch in range(world_config['TRAIN_epochs']):
    #     start = time.time()
    #     if epoch % 10 == 0:
    #         cprint('[TEST]')
    #         Procedure.Test(dataset, recModel, epoch, w, config['multicore'])
    #     Procedure.Score_train_original(dataset, recModel, loss, epoch, Neg_k, w)
    # Procedure.Test(dataset, recModel, world_config['TRAIN_epochs'], w, config['multicore'])

    torch.save(recModel.state_dict(), weight_file)
finally:
    if world_config['tensorboard']:
        w.close()
