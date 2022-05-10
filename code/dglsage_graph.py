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

from dgl_graph import load_lastfm_graph

config = world.config
world_config = world.world_config
path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', world_config['dataset'])
dataset = LastFM(path=path)

from torch.utils.data import Dataset, IterableDataset


class UserItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph[user_type][item_type])[0]  # 用户-物品边类型
        self.item_to_user_etype = list(g.metagraph[item_type][user_type])[0]  # 物品-用户边类型
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.user_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(  # 沿着边找到评价当前电影的用户，所评价的其它电影
                self.g,
                heads,
                # 这里是选择的元路径
                metapath=[self.user_to_item_etype, self.item_to_user_etype])[0][:, 1]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.user_type), (self.batch_size,))
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]


def compact_and_copy(frontier, seeds):
    # 这个frontier为一个图，seeds为种子节点，代表目的节点
    """
    将同构图转化为块图
    将第一轮的dst节点和大的图frontier压缩成小图block
    设置block的seeds为output，其他为input
    :param frontier:
    :param seeds:
    :return:
    """
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        # print("col:",col)
        # print("data:",data)
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    # print(block)
    return block


class NodesGraphCollactor():
    def __init__(self, g, user_type, item_type, neighbor_very_layer=None):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        if neighbor_very_layer is None:
            neighbor_very_layer = [5, 1]
        self.neighbor_very_layer = neighbor_very_layer

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        heads = torch.Tensor(heads)
        tails = torch.Tensor(tails)
        neg_tails = torch.Tensor(neg_tails)
        pos_graph, neg_graph, blocks = self.sample_from_pair(heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sample_blocks(batch)
        return blocks

    def sample_from_pair(self, heads, pos_tails, neg_tails):
        pos_graph = dgl.heterograph({
            (self.user_type, self.item_type): (heads, pos_tails)
        })
        neg_graph = dgl.heterograph({
            (self.user_type, self.item_type): (heads, neg_tails)
        })
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph[self.user_type].data[dgl.NID]
        blocks = self.sample_blocks(seeds, heads, pos_tails, neg_tails)
        return pos_graph, neg_graph, blocks

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for n_neighbor in self.neighbor_very_layer:
            froniter = dgl.sampling.sample_neighbors(
                self.g,
                seeds,
                fanout=n_neighbor,
                edge_dir='in'
            )
            if heads is not None:
                eids = froniter.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_froniter = froniter
                    froniter = dgl.remove_edges(old_froniter, eids)
            block = compact_and_copy(froniter, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks


from dgl_graph import Light_Layer


class LightGCNSage(nn.Module):
    def __init__(self, num_user, num_item, embedding_dim, n_layer):
        super(LightGCNSage, self).__init__()
        self.n_layer = n_layer
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(Light_Layer())
        self.embedding_user = nn.Embedding(num_user, embedding_dim)
        self.embedding_item = nn.Embedding(num_item, embedding_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_dim.weight, std=0.1)

    def computer(self, blocks,h_user,h_item):
        embs = [torch.cat([h_user, h_item])]
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_user, h_item = layer(block, h_user, h_item)
            embs.append(torch.cat([h_user, h_item]))
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        h_user, h_item = torch.split(light_out, [self.num_user, self.num_item])
        return h_user, h_item
