'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import math
import pickle
from time import time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
# from sklearn.metrics import roc_auc_score
from cppimport import imp_from_filepath
from torch import optim
import world
from dataloader import BasicDataset
from model import PairWiseModel

import random

world_config = world.world_config
config = world.config
from os.path import join, dirname
import sys

try:
    # if world_config['dataset'] in ['lastfm', 'amazon-electronic']:
    #     raise Exception
    # 一定要先把该文件夹加入到系统路径中
    sys.path.append(world_config['CODE_PATH'] + '/sources')
    print(world_config['CODE_PATH'] + '/sources')
    path = join(dirname(__file__), "sources", "sampling.cpp")
    print(path)
    sampling = imp_from_filepath(path)
    # sampling=cppimport.imp(path)
    world.cprint('CPP extension loaded')
    sampling.seed(world_config['seed'])
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


# sample_ext=False
class BPRLoss:
    def __init__(self,
                 recmodel,
                 config: dict):
        self.model = recmodel
        # print(config['decay'])
        self.l2_normalize = config['decay']
        self.lr = config['lr']
        # print(recmodel)
        params = []
        params.extend(recmodel.parameters())
        # for gate in recmodel.gates:
        #     params.extend(gate.parameters())
        self.opt = optim.Adam(params, lr=self.lr)

    # 在这里进行梯度下降，传播运算
    def stageOne(self, users, pos, neg, gum_temp=None, hard=None):
        if gum_temp is not None:
            loss, reg_loss, gating_dist, embs = self.model.bpr_loss(users, pos, neg, gum_temp, hard)
        else:
            loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # print('user:{},time:{}', users.shape[0],time2 - time1)
        # print(reg_loss)
        reg_loss = reg_loss * self.l2_normalize
        loss = loss + reg_loss
        self.opt.zero_grad()
        loss.backward()
        # print('user:{},time:{}', users.shape[0], time2 - time1)
        self.opt.step()
        return loss.cpu().item(), loss.cpu().item() - reg_loss.cpu().item(), reg_loss.cpu().item()


class ScoreLoss:
    def __init__(self,
                 recmodel: PairWiseModel,
                 config: dict):
        self.model = recmodel
        # self.l2_normalize = config['l2_normalize']
        self.lr = config['lr']
        self.weight_decay = config['decay']
        # 这里不要在加上weight_Decay了
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # print(list(recmodel.parameters()))
        # print(recmodel.ws)
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr, )
        # self.opt1 = optim.Adam(recmodel.ws, lr=self.lr)

    def stageOne(self, users, pos, neg, score):
        loss1, loss2, reg_loss = self.model.loss(users, pos, neg, score)
        regulation_loss = self.weight_decay * reg_loss
        loss = loss1 + loss2 + regulation_loss
        # self.opt1.zero_grad()
        self.opt.zero_grad()
        loss.backward()
        # self.opt1.step()
        self.opt.step()
        return loss.cpu().item(), loss1.cpu().item(), loss2.cpu().item(), regulation_loss.cpu().item()


def UniformSample_original(dataset, neg_ratio=1, epoch=None, score=False):
    dataset: BasicDataset
    allPos = dataset.allPos
    # print(allPos)
    start = time()
    try:
        # print('n_item:{}'.format(dataset.n_users))
        # print('m_item:{}'.format(dataset.m_items))
        # print('train_data_size:{}'.format(dataset.trainDataSize))
        if not score:
            S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                         dataset.trainDataSize, allPos, neg_ratio)
        else:
            allPosScore = dataset.allPosScores
            if config['sample_neg'] == 0:
                S = sampling.sample_negative_score(dataset.n_users, dataset.m_items,
                                                   dataset.trainDataSize, allPos, allPosScore, neg_ratio)
            else:
                sum_prob = dataset.getSampleSumRate()
                prob_list = dataset.getSampleProbRate()
                if epoch is not None:
                    # pass
                    prob_list = np.power(prob_list, 0.5 / (epoch / 10 + 1))
                    # prob_list = np.power(prob_list, 0.5)
                S = sampling.sample_negative_score_prob(dataset.n_users, dataset.m_items,
                                                        dataset.trainDataSize, allPos, allPosScore, prob_list,
                                                        neg_ratio)
    except:
        print('sampling in python')
        S = UniformSample_original_python(dataset, score=score)
    return S


def UniformSample_original_python(dataset, score=False):
    """
    the original impliment of BPR Sampling in LightGCN
    进行采用的过程，通过采样，得到正类样本和负类样本
    :return:
        np.array
    """
    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    if not score:
        for i, user in enumerate(users):
            start = time()
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            sample_time2 += time() - start
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
            end = time()
            sample_time1 += end - start
    else:
        allPosScores = dataset.allPosScores
        for i, user in enumerate(users):
            start = time()
            posForUser = allPos[user]
            posForUserScore = allPosScores[user]
            if len(posForUser) == 0:
                continue
            sample_time2 += time() - start
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            posScore = posForUserScore[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem, posScore])
            end = time()
            sample_time1 += end - start
    total = time() - total_start
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    # 这个是模型保存的地点
    file = ''
    if world_config['model_name'] == 'mf':
        file = f"mf-{world_config['dataset']}-{world_config['comment']}-{config['latent_dim_rec']}.pth.tar"

    elif world_config['model_name'] in {'lgn', 'ngcf', 'neumf', 'cmn', 'cf_mo', 'dhcf', 'bpr_cfig', 'cf_smp', 'dgcf',
                                        'drop_cf_mo','cf_ssl'}:
        file = "model_name-{}-dataset-{}-comment-{}-n_layers-{}-latent_dim-{}-delete_{}.pth.tar".format(world_config['model_name'], world_config['dataset'],
                                                  world_config['comment'], config['n_layers'],
                                                  config['latent_dim_rec'], config['delete_user'])
    return world_config['FILE_PATH'] + '/' + file


# return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', config['batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def preprocess_graph(adj, c=1):
    """ process the graph
        * options:
        normalization of augmented adjacency matrix
        formulation from convolutional filter
        normalized graph laplacian
    Parameters
    ----------
    adj: a sparse matrix represents the adjacency matrix
    Returns
    -------
    adj_normalized: a sparse matrix represents the normalized laplacian
        matrix
    """
    _adj = adj + c * sp.eye(adj.shape[0])  # Sparse matrix with ones on diagonal 产生对角矩阵
    # _D = sp.diags(_dseq)
    return normalize_graph(_adj, mode=2)


def normalize_graph(mat, mode=0):
    if mode == 0:
        _dseq = mat.sum(1).A1  # 按行求和后拉直
        # _dseq[_dseq == 0] = 1
        _dseq = np.power(_dseq, -0.5)
        _dseq[np.isinf(_dseq)] = 0.
        _dseq[np.isnan(_dseq)] = 0.
        _D_half1 = sp.diags(_dseq)  # 开平方构成对角矩阵
        # _D_half[sp]
        adj_normalized = _D_half1 @ mat @ _D_half1  # 矩阵乘法
    elif mode == 1:
        _dseq = mat.sum(1).A1  # 按行求和后拉直
        # _dseq[_dseq == 0] = 1
        _dseq = np.power(_dseq, -1)
        _dseq[np.isinf(_dseq)] = 0.
        _dseq[np.isnan(_dseq)] = 0.
        _D_half1 = sp.diags(_dseq)  # 开平方构成对角矩阵
        # _D_half[sp]
        adj_normalized = _D_half1 @ mat  # 矩阵乘法
    else:
        _dseq = mat.sum(1).A1  # 按行求和后拉直
        # _dseq[_dseq == 0] = 1
        _dseq = np.power(_dseq, -0.5)
        _dseq[np.isinf(_dseq)] = 0.
        _dseq[np.isnan(_dseq)] = 0.
        _D_half1 = sp.diags(_dseq)  # 开平方构成对角矩阵
        _dseq = mat.sum(0).A1  # 按列求和后拉直
        # _dseq[_dseq == 0] = 1
        _dseq = np.power(_dseq, -0.5)
        _dseq[np.isinf(_dseq)] = 0.
        _dseq[np.isnan(_dseq)] = 0.
        _D_half2 = sp.diags(_dseq)  # 开平方构成对角矩阵
        # _D_half[sp]
        adj_normalized = _D_half1 @ mat @ _D_half2  # 矩阵乘法
    return adj_normalized.tocsr()


def normalize_tensor_graph(graph: torch.sparse.Tensor, mode=0):
    m = graph.shape[0]
    D_col_indices = torch.Tensor(
        [list(range(m)), list(range(m))]).long()
    n = graph.shape[1]
    D_row_indices = torch.Tensor(
        [list(range(n)), list(range(n))]).long()
    if mode == 0:
        D_col_scores = (1. / torch.sqrt(torch.sparse.sum(graph, dim=1).to_dense()))  # 这里要去掉inf和nan
        D_col_scores[torch.isnan(D_col_scores)] = 0.
        D_col_scores[torch.isinf(D_col_scores)] = 0.
        D_col_tensor = torch.sparse.FloatTensor(D_col_indices, D_col_scores, graph.shape)
        graph = torch.sparse.mm(D_col_tensor, graph)
        graph = torch.sparse.mm(graph, D_col_tensor)
    if mode == 1:
        D_col_scores = (1. / torch.sparse.sum(graph, dim=1).to_dense())  # 这里要去掉inf和nan
        D_col_scores[torch.isnan(D_col_scores)] = 0.
        D_col_scores[torch.isinf(D_col_scores)] = 0.
        D_col_tensor = torch.sparse.FloatTensor(D_col_indices, D_col_scores, graph.shape)
        graph = torch.sparse.mm(D_col_tensor, graph)
    if mode == 2:
        # print(D)
        D_col_scores = (1. / torch.sqrt(torch.sparse.sum(graph, dim=1).to_dense()))  # 这里要去掉inf和nan
        D_row_scores = (1. / torch.sqrt(torch.sparse.sum(graph, dim=0).to_dense()))
        # print('D_i_col_scores inf',torch.sum(torch.isinf(D_i_col_scores).long()))
        D_col_scores[torch.isnan(D_col_scores)] = 0.
        D_col_scores[torch.isinf(D_col_scores)] = 0.
        D_row_scores[torch.isnan(D_row_scores)] = 0.
        D_row_scores[torch.isinf(D_row_scores)] = 0.
        D_col_tensor = torch.sparse.FloatTensor(D_col_indices, D_col_scores, graph.shape)
        D_row_tensor = torch.sparse.FloatTensor(D_row_indices, D_row_scores, graph.shape)
        graph = torch.sparse.mm(D_col_tensor, graph)
        graph = torch.sparse.mm(graph, D_row_tensor)
    # graph = graph.coalesce().to(world_config['device'])
    return graph


def construct_distance_matrix(UserItemNet, user=True):
    ## todo 构建距离的方法
    if user:
        if config['distance_measure'] == 'occurrence':
            mat = UserItemNet.dot(UserItemNet.transpose())
        elif config['distance_measure'] == 'cosine_similarity':
            mat = UserItemNet.dot(UserItemNet.transpose())
            sum0 = mat.sum(1).A1
            sum1 = mat.sum(0).A1
            sqrt0 = np.power(sum0, -0.5)
            sqrt0[np.isinf(sqrt0)] = 0.
            sqrt1 = np.power(sum1, -0.5)
            mat = sp.diags(sqrt0) @ mat @ sp.diags(sqrt1)
            mat = mat.tocsr()
        elif config['distance_measure'] == 'inverted':
            num_user, num_item = UserItemNet.shape
            W = defaultdict(lambda: 0)
            for j in range(num_item):
                # print('{}:{}'.format(j, num_item))
                if j % (num_item // 5) == 0:
                    print('finished:{}/5'.format(j // (num_item // 5)))
                user_list = UserItemNet[:, j].nonzero()[0]  # 与该物品交互过的用户集合
                # print(user_list)
                w = 10 / math.log(1.1 + len(user_list))
                # w = 1 / (1 + len(user_list))
                for u in user_list:
                    for v in user_list:
                        if u == v:
                            continue
                        W[u, v] += w
                        W[v, u] += w
            idx1 = [u for u, v in W.keys()]
            idx2 = [v for u, v in W.keys()]
            data = list(W.values())
            data = np.array(data)
            # print(data.shape)
            index = [np.array(idx1), np.array(idx2)]
            mat = sp.csr_matrix((data, index),
                                shape=(num_user, num_user))
            # print("mat", len(sorted(mat[0].nonzero()[1])))
            # print("UserItemNet.dot(UserItemNet.transpose()", len(sorted(UserItemNet.dot(UserItemNet.transpose())[0].nonzero()[1])))
    else:
        if config['distance_measure'] == 'occurrence':
            mat = (UserItemNet.transpose()).dot(UserItemNet)
        elif config['distance_measure'] == 'cosine_similarity':
            mat = (UserItemNet.transpose()).dot(UserItemNet)
            sum0 = mat.sum(1).A1
            sum1 = mat.sum(0).A1
            sqrt0 = np.power(sum0, -0.5)
            sqrt0[np.isinf(sqrt0)] = 0.
            sqrt1 = np.power(sum1, -0.5)
            mat = sp.diags(sqrt0) @ mat @ sp.diags(sqrt1)
            mat = mat.tocsr()

        elif config['distance_measure'] == 'inverted':
            num_user, num_item = UserItemNet.shape
            W = defaultdict(lambda: 0)
            for u in range(num_user):
                if u % (num_user // 5) == 0:
                    print('finished:{}/5'.format(u // (num_user // 5)))
                item_list = UserItemNet[u].nonzero()[1]
                # print(item_list)
                w = 10 / math.log(1.1 + len(item_list))
                # w = 1 / (1 + len(item_list))
                for i in item_list:
                    for j in item_list:
                        if i == j:
                            continue
                        W[i, j] += w
                        W[j, i] += w
            idx1 = [u for u, v in W.keys()]
            idx2 = [v for u, v in W.keys()]
            data = list(W.values())
            data = np.array(data)
            index = [np.array(idx1), np.array(idx2)]
            mat = sp.csr_matrix((data, index),
                                shape=(num_item, num_item))
            # print(mat)
    return mat


def construct_similar_graph(mat, num=8):
    """
    构建相似性图的方法，由于mat是稀疏矩阵，转化为稠密矩阵效果可能不好
    :param mat:
    :param num:
    :return:
    """
    n = mat.shape[0]
    mat = sp.lil_matrix(mat)
    mat[np.arange(n), np.arange(n)] = 0
    mat = sp.csr_matrix(mat)
    idxs1, idxs2 = [], []
    res_data = []
    for i in range(n):
        # print("{}:{}".format(i, n))
        idxes = mat[i].nonzero()[1]
        data = mat[i].data
        list1 = list(zip(idxes, data))
        list1.sort(key=lambda x: x[1])
        list1 = list1[-min(len(list1), num):]
        for j in range(len(list1)):
            idxs1.append(i)
            idxs2.append(list1[j][0])
            res_data.append(list1[j][1])
        if i % (n // 5) == 0:
            print('finished:{}/5'.format(i // (n // 5)))
    res_data = np.array(res_data)
    index = [np.array(idxs1), np.array(idxs2)]
    mat = sp.coo_matrix((res_data, index),
                        shape=(n, n))
    # mat = preprocess_graph(mat, 0)
    mat = normalize_graph(mat, 0)
    return mat


def construct_similar_third_graph(mat, num=8):
    m = mat.shape[0]
    n = mat.shape[1]
    mat = sp.csr_matrix(mat)
    idxs1, idxs2 = [], []
    res_data = []
    for i in range(m):
        # print("{}:{}".format(i, n))
        idxes = mat[i].nonzero()[1]
        data = mat[i].data
        list1 = list(zip(idxes, data))
        list1.sort(key=lambda x: x[1])
        list1 = list1[-min(len(list1), num):]
        for j in range(len(list1)):
            idxs1.append(i)
            idxs2.append(list1[j][0])
            res_data.append(list1[j][1])
        if i % (m // 5) == 0:
            print('finished:{}/5'.format(i // (m // 5)))
    res_data = np.array(res_data)
    index = [np.array(idxs1), np.array(idxs2)]
    mat = sp.coo_matrix((res_data, index),
                        shape=(m, n))
    # mat = preprocess_graph(mat, 0)
    mat = normalize_graph(mat, 2)
    return mat


def preprocess_adjacency_graph(graph, num_user, num_item):
    """
    进行处理一个邻接关系图
    进行随机游走，在随机游走经过的物品节点中寻找topK个用户
    :param graph: 用户-物品邻接关系图,用稀疏矩阵来存储
    :param topK: 针对每个用户，寻找关系最近的K个物品
    :return:
    """
    import os
    top_k = config['adj_top_k']
    path = join(dirname(os.path.dirname(__file__)), 'data', world_config['dataset'])
    idx1_path = "{0}{1}".format(path,
                                '/top_{}_idx1'.format(top_k))
    idx2_path = "{0}{1}".format(path,
                                '/top_{}_idx2'.format(top_k))
    data_path = "{0}{1}".format(path,
                                '/top_{}_data'.format(top_k))
    if os.path.exists(idx1_path):
    # if False:
        print(idx1_path)
        with open(idx1_path, 'rb') as f:
            idxs1 = pickle.load(f)
        with open(idx2_path, 'rb') as f:
            idxs2 = pickle.load(f)
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        num_path = config['num_path']  # 每个用户随机游走路径数量
        path_length = config['path_length']  # 随机游走路径长度
        top_k = config['adj_top_k']
        alpha = config['restart_alpha']
        G = {}
        idxs1, idxs2 = [], []
        user2itemset = defaultdict(lambda: set())
        for i in range(num_user):
            adj_user = (num_user + graph[i].nonzero()[1]).tolist()
            G[i] = adj_user
            user2itemset[i] = set(adj_user)

        for j in range(num_item):
            adj_item = graph[:, j].nonzero()[0].tolist()
            G[j + num_user] = adj_item
        # print("G[1234]", [2345])
        # print(G)
        data = []
        for i in range(num_user):
            if i % (num_user // 5) == 0:
                print('deal {}/5 users'.format(i // (num_user // 5)))
            path = random_walk_path(G, num_path * path_length, alpha=alpha, start=i)
            count_list = [(a, b) for a, b in list(Counter(path).items()) if a >= num_user]  # and a in user2itemset[i]]
            count_list.sort(key=lambda x: -x[1])  # 取最大的K个
            print("count_list: ", count_list)
            top_k_list = [a for (a, b) in count_list[:min(top_k, len(count_list), )]]
            top_k_data = [b for (a, b) in count_list[:min(top_k, len(count_list), )]]
            print('---------')
            print('{}的邻居个数:{}'.format(i, len(G[i])))
            print("{}的topK长度:{}".format(i, len(top_k_list)))
            print("topK中邻居个数:{}".format(len(set(G[i]).intersection(set(top_k_list)))))
            print("---------")
            idxs1.extend([i] * len(top_k_list))
            idxs2.extend(top_k_list)
            data.extend(top_k_data)
            idxs2.extend([i] * len(top_k_list))
            idxs1.extend(top_k_list)
            data.extend(top_k_data)
        with open(idx1_path, 'wb') as f:
            pickle.dump(idxs1, f)
        with open(idx2_path, 'wb') as f:
            pickle.dump(idxs2, f)
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
    # print(data)
    idxs1 = torch.LongTensor(idxs1)
    idxs2 = torch.LongTensor(idxs2)
    data = torch.FloatTensor(data)
    # print(idxs1.shape)
    # print(idxs2.shape)
    index = torch.stack([idxs1, idxs2])
    top_k_graph = torch.sparse.FloatTensor(index, data,  # torch.ones(index.shape[1]),
                                           torch.Size([num_user + num_item, num_user + num_item]))
    return top_k_graph


def random_walk_path(G, path_length, alpha=0., start=0):
    """
    #获得从start开始的一条随机游走的路径
    :param G: 原始图，为邻接表结构
    :param path_length: 路径的长度
    :param alpha: 返回原始起点的概率
    :param start: 起始节点
    :return: 一条路径
    """
    # if start:
    #     path = [start]
    # else:
    #     path = [random.choice(list(G.keys()))]
    path = [start]
    while len(path) < path_length:
        cur = path[-1]
        if len(G[cur]) > 0:
            if random.random() >= alpha:
                # print(G[cur])
                path.append(random.choice(G[cur]))
            else:
                path.append(path[0])
        else:
            path.append(path[0])
    # print(path)
    return path


def preprocess_topk_score_graph(graph, num_user, num_item):
    """
    根据分数寻找topK个邻居，选择指标为分数大小
    :param graph:
    :param num_user:
    :param num_item:
    :return:
    """
    idxs1, idxs2 = [], []
    top_k = config['adj_top_k']
    for i in range(num_user):
        adj_user = (num_user + graph[i].nonzero()[1]).tolist()
        adj_score_user = graph[i].data
        tup_list = list(zip(adj_user, adj_score_user))
        tup_list.sort(key=lambda x: -x[1])
        # print("邻居个数:{},删选后个数:{}".format(len(tup_list),top_k))
        tup_list = tup_list[:min(top_k, len(tup_list))]
        for tup in tup_list:
            idxs1.append(i)
            idxs2.append(tup[0])
            idxs2.append(i)
            idxs1.append(tup[0])
    idx1 = torch.LongTensor(idxs1)
    idx2 = torch.LongTensor(idxs2)
    index = torch.stack([idx1, idx2])
    top_k_graph = torch.sparse.FloatTensor(index, torch.ones(index.shape[1]),
                                           torch.Size([num_user + num_item, num_item + num_user]))
    return top_k_graph


def cnt2prob(user2cnt):
    a = 1e-4  # config['a']
    user2prob = (np.sqrt(user2cnt / a) + 1) * (a / user2cnt)
    user2prob = user2prob / np.sum(user2prob)
    return user2prob


def preprocess_random_select_graph(graph, num_user, num_item):
    top_k = config['adj_top_k']
    # user2cnt = np.array(list(Counter(graph.nonzero()[0]).values()))  # user的计数
    item2cnt = Counter(graph.nonzero()[1])
    item2cnt = np.array([item2cnt[i] for i in range(num_item)])
    # print(list(Counter(graph.nonzero()[1]).values()))
    a = 1e-3  # config['a']，超参数，需要后需设置
    # user2prob = (np.sqrt(user2cnt / a) + 1) * (a / user2cnt)
    idxs1, idxs2 = [], []
    for i in range(num_user):
        adj_items = (graph[i].nonzero()[1]).tolist()
        # print(adj_items)
        adj_items_cnt = item2cnt[adj_items]
        adj_items_prob = cnt2prob(adj_items_cnt)
        # print(adj_items_prob)
        if len(adj_items) == 0:
            print('user :{} size is 0'.format(i))
            continue
        adj_items_select = np.random.choice(adj_items, size=min(top_k, len(adj_items)), replace=False, p=adj_items_prob)
        for j in adj_items_select.tolist():
            idxs1.append(i)
            idxs2.append(j + num_user)
            idxs1.append(j + num_user)
            idxs2.append(i)
    idx1 = torch.LongTensor(idxs1)
    idx2 = torch.LongTensor(idxs2)
    index = torch.stack([idx1, idx2])
    top_k_graph = torch.sparse.FloatTensor(index, torch.ones(index.shape[1]),
                                           torch.Size([num_user + num_item, num_item + num_user]))
    return top_k_graph


def transform_score(score: np.ndarray):
    """
    对评分进行划分，划分为1-5
    :param score:
    :return:
    """
    data = pd.DataFrame({'score': score})
    scores1 = score.copy()
    scores1 = np.sort(scores1)
    num_item = len(scores1)
    bin = num_item // 5

    def cut(val):
        if val <= scores1[bin]:
            return 1
        elif val <= scores1[bin * 2]:
            return 2
        elif val <= scores1[bin * 3]:
            return 3
        elif val <= scores1[bin * 4]:
            return 4
        return 5

    data['new_score'] = data['score'].apply(cut)
    return data['new_score']


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def sample_cor_samples(n_users, n_items, cor_batch_size):
    """
        We have to sample some embedded representations out of all nodes.
        Because we have no way to store cor-distance for each pair.
    """
    cor_users = random.sample(list(range(n_users)), cor_batch_size)
    cor_items = random.sample(list(range(n_items)), cor_batch_size)
    cor_users = torch.LongTensor(cor_users)
    cor_items = torch.LongTensor(cor_items)
    return cor_users, cor_items
# ====================end Metrics=============================
# =========================================================
