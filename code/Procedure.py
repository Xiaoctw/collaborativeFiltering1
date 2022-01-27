from __future__ import division
from __future__ import print_function

import random

import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer, sample_cor_samples
from time import time
from tqdm import tqdm
import model
import multiprocessing
import models

CORES = multiprocessing.cpu_count() // 2
config = world.config
world_config = world.world_config


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    with timer(name="Sample"):
        #  print('start sampling')
        S = utils.UniformSample_original(dataset)
    # print('finish sampling')
    # 这个说明每个轮次的负样本都不太一样
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users = users.to(world_config['device'])
    posItems = posItems.to(world_config['device'])
    negItems = negItems.to(world_config['device'])
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config['batch_size'] + 1
    aver_loss = 0.
    bpr_loss = 0.
    reg_loss = 0.
    time1 = time()
    #  print('start batch')
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=config['batch_size'])):
        # 在这里梯度下降
        cri, bpr_batch, reg_batch = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        bpr_loss += bpr_batch
        reg_loss += reg_batch
        if world_config['tensorboard']:
            # 这三个参数分别表示 保存图的名称，Y轴数据，X轴数据
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / config['batch_size']) + batch_i)
    time2 = time()
    aver_loss = aver_loss / total_batch
    bpr_loss = bpr_loss / total_batch
    reg_loss = reg_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    output_information = f"loss{aver_loss:.3f}-{time_info}"
    print('EPOCH:{}/{},bpr_loss:{:.4f},reg_loss:{:.4f},avg_loss:{:.4f},time_info:{},time:{:.4f}'.format(epoch + 1,
                                                                                                        world_config[
                                                                                                            'TRAIN_epochs'],
                                                                                                        bpr_loss,
                                                                                                        reg_loss,
                                                                                                        aver_loss,
                                                                                                        time_info,
                                                                                                        time2 - time1))
    # print(f'EPOCH[{epoch + 1}/{world_config["TRAIN_epochs"]}] {output_information}')
    return output_information


def Score_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.ScoreLoss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset, score=True, epoch=epoch)
    # 这个说明每个轮次的负样本都不太一样
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    posScores = torch.Tensor(S[:, 3]).float()
    # print(users)
    # print(posItems)
    # print(negItems)
    # print(posScores)
    users = users.to(world_config['device'])
    posItems = posItems.to(world_config['device'])
    negItems = negItems.to(world_config['device'])
    posScores = posScores.to(world_config['device'])
    users, posItems, negItems, posScores = utils.shuffle(users, posItems, negItems, posScores)
    # users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config['batch_size'] + 1
    aver_loss = 0.
    _loss1 = 0.
    _loss2 = 0
    reg_loss = 0.
    time1 = time()
    #  print('start batch')
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg, batch_score
          )) in enumerate(utils.minibatch(users,
                                          posItems,
                                          negItems,
                                          posScores,
                                          batch_size=config['batch_size'])):
        # 在这里梯度下降
        # show_pred=False
        # if batch_i==users.shape[0]//config['batch_size']-1 and random.random()<0.2:
        #     show_pred=True
        cri, l1, l2, reg_batch = loss.stageOne(batch_users, batch_pos, batch_neg, batch_score)
        # cri, l1, l2, reg_batch = loss.stageOne(batch_users, batch_pos, batch_neg, 0)
        # print('cri:{},bpr_loss:{},reg_loss:{}'.format(cri,bpr_batch,reg_batch))
        aver_loss += cri
        _loss1 += l1
        _loss2 += l2
        reg_loss += reg_batch
        if world_config['tensorboard']:
            # 这三个参数分别表示 保存图的名称，Y轴数据，X轴数据
            w.add_scalar(f'Loss/BPR', cri, epoch * int(len(users) / config['batch_size']) + batch_i)
    time2 = time()
    aver_loss = aver_loss / total_batch
    _loss1 = _loss1 / total_batch
    _loss2 = _loss2 / total_batch
    reg_loss = reg_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    output_information = f"loss{aver_loss:.3f}-{time_info}"
    print('EPOCH:{}/{},crs loss:{:.4f},mse loss:{:.4f},reg_loss:{:.4f},avg_loss:{:.4f},time_info:{},time:{:.4f}'.format(
        epoch + 1,
        world_config['TRAIN_epochs'],
        _loss1,
        _loss2,
        reg_loss,
        aver_loss,
        time_info,
        time2 - time1))
    # print(f'EPOCH[{epoch + 1}/{world_config["TRAIN_epochs"]}] {output_information}')
    return output_information


def SSL_train_original(dataset, recommend_model:models.CF_SSL, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    loss: utils.ScoreLoss = loss_class
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset, score=True, epoch=epoch)
    # 这个说明每个轮次的负样本都不太一样
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    posScores = torch.Tensor(S[:, 3]).float()
    users = users.to(world_config['device'])
    posItems = posItems.to(world_config['device'])
    negItems = negItems.to(world_config['device'])
    posScores = posScores.to(world_config['device'])
    users, posItems, negItems, posScores = utils.shuffle(users, posItems, negItems, posScores)
    # users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // config['batch_size'] + 1
    aver_loss = 0.
    _loss1 = 0.
    _loss2 = 0
    reg_loss = 0.
    time1 = time()
    #  print('start batch')
    Recmodel.create_adj_mat() #每次迭代前都要重新构建图
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg, batch_score
          )) in enumerate(utils.minibatch(users,
                                          posItems,
                                          negItems,
                                          posScores,
                                          batch_size=config['batch_size'])):
        cri, l1, l2, reg_batch = loss.stageOne(batch_users, batch_pos, batch_neg, batch_score)
        aver_loss += cri
        _loss1 += l1
        _loss2 += l2
        reg_loss += reg_batch
        if world_config['tensorboard']:
            w.add_scalar(f'Loss/BPR', cri, epoch * int(len(users) / config['batch_size']) + batch_i)
    time2 = time()
    aver_loss = aver_loss / total_batch
    _loss1 = _loss1 / total_batch
    _loss2 = _loss2 / total_batch
    reg_loss = reg_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    output_information = f"loss{aver_loss:.3f}-{time_info}"
    print('EPOCH:{}/{},bpr loss:{:.4f},ssl loss:{:.4f},reg_loss:{:.4f},avg_loss:{:.4f},time_info:{},time:{:.4f}'.format(
        epoch + 1,
        world_config['TRAIN_epochs'],
        _loss1,
        _loss2,
        reg_loss,
        aver_loss,
        time_info,
        time2 - time1))
    return output_information


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world_config['topks']:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, w=None, multicore=0, gum_temp=None, hard=None):
    time1 = time()
    u_batch_size = config['test_u_batch_size']
    # dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    # Recmodel: model.PairWiseModel
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world_config['topks'])  # 一个列表
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world_config['topks'])),
               'recall': np.zeros(len(world_config['topks'])),
               'ndcg': np.zeros(len(world_config['topks']))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        # all_users, all_items = Recmodel.computer()
        # print(all_users)
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world_config['device'])
            if gum_temp != None:
                rating = Recmodel.getUsersRating(batch_users_gpu, gum_temp, hard)
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating  # 避免爆存，节省空间
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            print('pool begin')
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size / len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world_config['tensorboard']:
            w.add_scalars(f'Test/Recall@{world_config["topks"]}',
                          {str(world_config["topks"][i]): results['recall'][i] for i in
                           range(len(world_config["topks"]))}, epoch)
            w.add_scalars(f'Test/Precision@{world_config["topks"]}',
                          {str(world_config["topks"][i]): results['precision'][i] for i in
                           range(len(world_config["topks"]))}, epoch)
            w.add_scalars(f'Test/NDCG@{world_config["topks"]}',
                          {str(world_config["topks"][i]): results['ndcg'][i] for i in
                           range(len(world_config["topks"]))}, epoch)
        if multicore == 1:
            pool.close()
        #  print(results)
        time2 = time()
        print('recall:{},precision:{},ndcg:{},time={:.4f}'.format(results['recall'], results['precision'],
                                                                  results['ndcg'], time2 - time1))

        return results
