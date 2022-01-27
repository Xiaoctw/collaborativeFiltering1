import world
import utils
from world import cprint
import torch
import time
import Procedure
import os
from warnings import simplefilter


#这个和main其实是一样的
simplefilter(action="ignore", category=FutureWarning)

world_config = world.world_config
config = world.config
if not os.path.exists(world_config['FILE_PATH']):
    os.makedirs(world_config['FILE_PATH'], exist_ok=True)
# ==============================
utils.set_seed(world_config['seed'])
print(">>SEED:", world_config['seed'])
# ==============================
print(world_config['device'])
import register
from register import dataset

# 构建一个模型
Recmodel = register.MODELS[world_config['model_name']](config, dataset)
Recmodel = Recmodel.to(world_config['device'])
# loss = utils.BPRLoss(Recmodel, config)
loss = register.LOSSES[world_config['loss']](Recmodel,config)
# print(Recmodel)
# 模型参数保存的地点
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world_config['LOAD']:
    try:
        # 导入现有模型
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world_config['tensorboard']:
    summary_path = world_config['BOARD_PATH'] + '/' + (
            time.strftime("%m-%d-%Hh%Mm%Ss-") + world_config['dataset'] + '-' + world_config['comment'])
    from tensorboardX import SummaryWriter

    w: SummaryWriter = SummaryWriter(str(summary_path))
# join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    world.cprint("not enable tensorflow board")

try:
    for epoch in range(world_config['TRAIN_epochs']):
        start = time.time()
        if epoch % 10 == 0:
            world.cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, config['multicore'])
        Procedure.BPR_train_original(dataset, Recmodel, loss, epoch, neg_k=Neg_k, w=w)
    Procedure.Test(dataset, Recmodel, world_config['TRAIN_epochs'], w, config['multicore'])
    # print(f'EPOCH[{epoch + 1}/{world_config["TRAIN_epochs"]}] {output_information}')
    # 每一轮迭代都会保存一次，没有什么必要
    torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world_config['tensorboard']:
        w.close()
