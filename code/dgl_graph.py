import time
import os
import torch
import numpy as np
import dgl
import pandas as pd
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
import scipy.sparse as sp
import world
import utils
import dataloader
from dataloader import BasicDataset
from lastfm_dataloader import LastFM
import Procedure
from collections import defaultdict

config = world.config
world_config = world.world_config
path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', world_config['dataset'])
dataset = LastFM(path=path)


def load_lastfm_graph():
    trainData = np.array(dataset.trainData)
    trainUser = dataset.trainUser
    trainItem = dataset.trainItem
    num_user = max(dataset.unique_user) + 1
    num_item = max(dataset.unique_item) + 1
    user2cnt = defaultdict(lambda: 0)
    item2cnt = defaultdict(lambda: 0)
    print('trainData', trainData.shape)
    for i in range(trainData.shape[0]):
        user2cnt[trainData[i][0]] += 1
        item2cnt[trainData[i][1]] += 1
    buyScore = torch.FloatTensor([1 / item2cnt[item] for item in trainData[:, 1].tolist()])
    boughtScore = torch.FloatTensor([1 / user2cnt[user] for user in trainData[:, 0].tolist()])
    trainUser = torch.LongTensor(trainUser)
    trainItem = torch.LongTensor(trainItem)

    neighbor_num = config['neighbor_num']
    dist_mea = config['distance_measure']
    user_path = path + '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea, neighbor_num)
    item_path = path + '/item_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea, neighbor_num)
    user_mat, item_mat = None, None
    if os.path.exists(user_path) and os.path.exists(item_path):
        user_mat = sp.load_npz(user_path)
        item_mat = sp.load_npz(item_path)
    else:
        assert False
    user_sim_score = torch.FloatTensor(user_mat.data)
    item_sim_score = torch.FloatTensor(item_mat.data)
    # 构建这么一个图
    g = dgl.heterograph(data_dict={
        ("user", "buy", 'item'): (trainUser, trainItem),
        ("item", "bought", "user"): (trainItem, trainUser),
        ("user", 'user_sim', "user"): user_mat,
        ("item", "item_sim", "item"): item_mat
    }, num_nodes_dict={'user': num_user, 'item': num_item})
    g.edges['buy'].data['w'] = buyScore
    g.edges['bought'].data['w'] = boughtScore
    g.edges['user_sim'].data['w'] = user_sim_score
    g.edges['item_sim'].data['w'] = item_sim_score
    print(buyScore)
    print(boughtScore)
    return g, num_user, num_item


# def add_sim_adj_edges(dataset: BasicDataset, g:dgl.DGLHeteroGraph):
#     neighbor_num = config['neighbor_num']
#     dist_mea = config['distance_measure']
#     user_path = path + '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea,neighbor_num)
#     item_path = path + '/item_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea,neighbor_num)
#     user_mat, item_mat = None, None
#     if os.path.exists(user_path) and os.path.exists(item_path):
#         user_mat = sp.load_npz(user_path)
#         item_mat = sp.load_npz(item_path)
#     else:
#         assert False
#     g.add_edges(u=)

class Light_Layer(nn.Module):
    def __init__(self, ):
        """
        无参数，LightGCN的layer
        """
        super(Light_Layer, self).__init__()

    def forward(self, g, h_user, h_item):
        with g.local_scope():
            g.nodes['user'].data['h'] = h_user
            g.nodes['item'].data['h'] = h_item
            # g.edges['buy'].data['w'] = w_buy
            # g.edges['bought'].data['w'] = w_bought
            funcs = {
                'buy': (fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h')),
                'bought': (fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h')),
            }
            g.multi_update_all(funcs, 'mean')
            return g.nodes['user'].data['h'], g.nodes['item'].data['h']


class Sim_Layer(nn.Module):
    def __init__(self):
        super(Sim_Layer, self).__init__()

    def forward(self, g, h_user, h_item):
        with g.local_scope():
            g.nodes['user'].data['h'] = h_user
            g.nodes['item'].data['h'] = h_item
            funcs = {
                'user_sim': (fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h')),
                'item_sim': (fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            }
            g.multi_update_all(funcs, 'mean')
            return g.nodes['user'].data['h'], g.nodes['item'].data['h']


class LightGCN_dgl(nn.Module):
    def __init__(self, graph, num_user, num_item, n_layer, embedding_dim):
        super(LightGCN_dgl, self).__init__()
        self.n_layer = n_layer
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.embedding_user = nn.Embedding(num_user, embedding_dim)
        self.embedding_item = nn.Embedding(num_item, embedding_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.layers = nn.ModuleList()
        self.graph = graph
        for _ in range(self.n_layer):
            self.layers.append(Light_Layer())

    def computer(self):
        g = self.graph
        h_user = self.embedding_user.weight
        h_item = self.embedding_item.weight
        embs = [torch.cat([h_user, h_item])]
        with g.local_scope():
            for i in range(len(self.layers)):
                h_user, h_item = self.layers[i](g, h_user, h_item)
                embs.append(torch.cat([h_user, h_item]))
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        h_user, h_item = torch.split(light_out, [self.num_user, self.num_item])
        return h_user, h_item

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        # 直接在这里进行矩阵乘法，最终分数越高，评分越高
        rating = torch.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss


class CLAGL_dgl(LightGCN_dgl):
    def __init__(self, graph, num_user, num_item, n_layer, embedding_dim):
        super(CLAGL_dgl, self).__init__(graph, num_user, num_item, n_layer, embedding_dim)
        self.light_layer = Light_Layer()
        self.sim_layer = Sim_Layer()

    def computer(self, ):
        g = self.graph
        h_user = self.embedding_user.weight
        h_item = self.embedding_item.weight
        embs = [torch.cat([h_user, h_item])]
        with g.local_scope():
            h_user1, h_item1 = self.light_layer(g, h_user, h_item)
            h_user2, h_item2 = self.sim_layer(g, h_user, h_item)
            embs.append(torch.cat([h_user1, h_item1]))
            embs.append(torch.cat([h_user2, h_item2]))
            if self.n_layer > 2:
                h_user3, h_item3 = self.light_layer(g, h_user2, h_item2)
                embs.append(torch.cat([h_user3, h_item3]))
            if self.n_layer > 3:
                h_user4, h_item4 = self.sim_layer(g, h_user2, h_item2)
                embs.append(torch.cat([h_user4, h_item4]))

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        h_user, h_item = torch.split(light_out, [self.num_user, self.num_item])
        return h_user, h_item



g, num_user, num_item = load_lastfm_graph()
print(g)
print(g.edges['user_sim'])
n_layer = config['n_layers']
embedding_dim = config['latent_dim_rec']
epochs = world_config['TRAIN_epochs']
# rec_model = LightGCN_dgl(g, num_user, num_item, n_layer, embedding_dim)
rec_model = CLAGL_dgl(g, num_user, num_item, n_layer, embedding_dim)
rec_model = rec_model.to(world_config['device'])
loss = utils.BPRLoss(rec_model, config)
g = g.to(world_config['device'])

for epoch in range(epochs):
    start = time.time()
    if epoch % 10 == 0:
        world.cprint('begin to test')
        Procedure.Test(dataset, rec_model, epoch, None, config['multicore'])
    Procedure.BPR_train_original(dataset, rec_model, loss, epoch, neg_k=1, w=None)
Procedure.Test(dataset, rec_model, epochs, None, config['multicore'])
