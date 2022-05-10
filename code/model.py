import math
import time

import torch
import torch.nn.functional as F
from torch import nn
import dataloader
import world
from dataloader import BasicDataset

config = world.config
world_config = world.world_config


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

    def getColdStartEmbedding(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    """
    简单的矩阵分解模型
    """

    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        # 嵌入层维度
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    # print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        # 对所有的物品均进行计算
        items_emb = self.embedding_item.weight
        users_emb = self.embedding_user(users)
        scores = torch.matmul(users_emb, items_emb.t())
        return torch.sigmoid(scores)

    # def bpr_loss(self, users, pos, neg):
    #     (users_emb, pos_emb, neg_emb,
    #      userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
    #     reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
    #                           posEmb0.norm(2).pow(2) +
    #                           negEmb0.norm(2).pow(2)) / float(len(users))
    #     pos_scores = torch.mul(users_emb, pos_emb)
    #     pos_scores = torch.sum(pos_scores, dim=1)
    #     neg_scores = torch.mul(users_emb, neg_emb)
    #     neg_scores = torch.sum(neg_scores, dim=1)
    #     loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    #     return loss, reg_loss

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)

    def computer(self):
        return self.embedding_user.weight, self.embedding_item.weight


class LightGCN(PairWiseModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.dropout = self.config['dropout']
        # print(self.latent_dim)
        # print(self.n_layers)
        # print(self.keep_prob)
        # print(self.A_split)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        # 采用没有初始化的模型参数
        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')

        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # 获得一个稀疏图
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        # x是一个稀疏图
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            # print('没有dropout')
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                # 怎么划分的就怎么合并上
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                # print('没有split')
                # 将drop后的图和嵌入向量相乘
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        # 拼接之后维度会多一维
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        # 取了均值
        light_out = torch.mean(embs, dim=1)
        # 获得用户和物品嵌入向量，在第0维，将数据划分为num_users和num_items个数据。
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getColdStartEmbedding(self, users):
        users_emb = torch.matmul(users, self.embedding_item.weight)
        return users_emb

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
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # print('loss:{}'.format(loss))
        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class NGCF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(NGCF, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        # 这里判断是否是分割的图
        self.A_split = self.config['A_split']
        self.norm = nn.BatchNorm1d(self.latent_dim)
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        for i in range(1, self.n_layers + 1):
            setattr(self, 'W1_{}'.format(i), nn.Linear(self.latent_dim, self.latent_dim))
            setattr(self, 'W2_{}'.format(i), nn.Linear(self.latent_dim, self.latent_dim))
            # nn.init.normal_(getattr(self, 'W1_{}'.format(i)).weight, std=0.1)
            # nn.init.normal_(getattr(self, 'W2_{}'.format(i)).weight,std=0.1)
            nn.init.xavier_uniform_(getattr(self, 'W1_{}'.format(i)).weight, gain=1)
            nn.init.xavier_uniform_(getattr(self, 'W2_{}'.format(i)).weight, gain=1)
        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        # 这个控制是否dropout,默认的话lgn是不dropout的，ngcf要dropout
        print('ngcf is already to go(dropout:{})'.format(self.config['dropout']))

    def __dropout_X(self, x, keep_prob):
        """
        :param x: 一个稀疏图
        :param keep_prob: 保留的概率
        :return:
        """
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        # 这里相当于是进行一步缩放
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            graph_self = []
            for g in self.graph:
                graph.append(self.__dropout_X(g, keep_prob))
            for g in self.graph_self:
                graph_self.append(self.__dropout_X(g, keep_prob))
        else:
            graph = self.__dropout_X(self.graph, keep_prob)
            graph_self = self.__dropout_X(self.graph_self, keep_prob)
        return graph, graph_self

    def computer(self):
        user_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([user_emb, items_emb])
        # embs = [all_emb]
        if self.config['dropout'] and self.training:
            print('droping')
            g_droped, g_droped_self = self.__dropout(self.keep_prob)
        else:
            g_droped, g_droped_self = self.graph, self.graph_self
        # 这里进行了一步修改，利用不加上self的图
        embs = [all_emb]
        for layer in range(1, self.n_layers + 1):
            if self.A_split:
                tem_emb = []
                for f in range(len(g_droped)):
                    # 这里用droped_self
                    E1 = torch.sparse.mm(g_droped_self[f], all_emb)
                    E1 = getattr(self, 'W1_{}'.format(layer))(E1)
                    # 这里用的不加self
                    E2 = torch.sparse.mm(g_droped[f], all_emb * all_emb)
                    E2 = getattr(self, 'W2_{}'.format(layer))(E2)
                    # E = self.norm(E1 + E2)
                    E = E1 + E2
                    # E = F.dropout(E, p=0.5)
                    tem_emb.append(F.leaky_relu(E, negative_slope=0.2))
                side_emb = torch.cat(tem_emb, dim=0)
                all_emb = side_emb
            else:
                # 这里用droped_self
                E1 = torch.sparse.mm(g_droped_self, all_emb)
                E1 = getattr(self, 'W1_{}'.format(layer))(E1)
                # 这里用的不加self
                E2 = torch.sparse.mm(g_droped, all_emb * all_emb)
                E2 = getattr(self, 'W2_{}'.format(layer))(E2)
                all_emb = E1 + E2
                all_emb = F.leaky_relu(all_emb, negative_slope=0.2)
        ngcf_out = all_emb  # 不进行压缩的话效果更好
        # ngcf_out = torch.cat(embs, dim=1)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        # 取了均值
        ngcf_out = torch.mean(embs, dim=1)
        users, items = torch.split(ngcf_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
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
        for i in range(1, self.n_layers + 1):
            mat1 = getattr(self, 'W1_{}'.format(i)).weight
            reg_loss += (1 / 2) * (mat1.norm(2).pow(2) / mat1.shape[0])
            mat2 = getattr(self, 'W2_{}'.format(i)).weight
            reg_loss += 0.5 * mat2.norm(2).pow(2) / mat2.shape[0]
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class NeuMF(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(NeuMF, self).__init__()
        self.config = config
        self.dataset: BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.keep_prob = self.config['keep_prob']
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim,
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim,
        )
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        self.layer_dims = [2 * self.latent_dim, self.latent_dim, self.latent_dim]
        # self.layer_dims = [ self.latent_dim, int(self.latent_dim), self.latent_dim // 2, 1]
        # self.drop = nn.Dropout(p=1 - self.keep_prob)
        # self.drop = nn.Dropout(p=1)
        for i in range(1, len(self.layer_dims)):
            setattr(self, 'lin_{}'.format(i), nn.Linear(self.layer_dims[i - 1], self.layer_dims[i]))
            setattr(self, 'bn_{}'.format(i), nn.BatchNorm1d(self.layer_dims[i]))
            nn.init.xavier_uniform_(getattr(self, 'lin_{}'.format(i)).weight, gain=1)
        self.comb_layers_dims = [2 * self.latent_dim, self.latent_dim, self.latent_dim // 2, 1]
        for i in range(1, len(self.comb_layers_dims)):
            setattr(self, 'comb_lin_{}'.format(i), nn.Linear(self.comb_layers_dims[i - 1], self.comb_layers_dims[i]))
            setattr(self, 'comb_bn_{}'.format(i), nn.BatchNorm1d(self.comb_layers_dims[i]))
            nn.init.xavier_uniform_(getattr(self, 'comb_lin_{}'.format(i)).weight, gain=1)
        print('NeuMF is already')

    def getUsersRating(self, users, all_users=None, all_items=None, sigmod=True):
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item.weight
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        # 每一个物品都参与到运算当中
        # users_emb = users_emb.unsqueeze(0)
        # items_emb = items_emb.unsqueeze(1)
        # mat = (users_emb * items_emb).reshape((-1, self.latent_dim))
        # print(users_emb.shape)
        users = users_emb.repeat(1, num_item).reshape(num_user * num_item, -1)
        items = items_emb.repeat(num_user, 1)
        # mat = torch.cat([users, items], dim=1)
        if sigmod:
            # 这里相当于是直接得到评分
            output = torch.sigmoid(self.helper((users, items)))
        else:
            output = self.helper((users, items))
        output = output.reshape(num_user, num_item)
        return output

    def helper(self, input):
        """
        将输入的用户和物品嵌入矩阵传入网络当中进行计算输出结果
        :param input:
        :return:
        """
        # print('-----------')
        users_emb, items_emb = input[0], input[1]
        # users_emb = users_emb.unsqueeze(0)
        # items_emb = items_emb.unsqueeze(1)
        # x = (users_emb * items_emb).reshape((-1, self.latent_dim))
        # x=users_emb*items_emb
        x1 = torch.cat([users_emb, items_emb], dim=1)
        for i in range(1, len(self.layer_dims)):
            x1 = getattr(self, 'lin_{}'.format(i))(x1)
            x1 = F.leaky_relu(x1, negative_slope=0.2)
            # if i < len(self.layer_dims) - 1:
            # x = getattr(self, 'bn_{}'.format(i))(x)
            # x1 = self.drop(x1)
        x2 = torch.mul(users_emb, items_emb)
        x = torch.cat([x1, x2], dim=1)
        for i in range(1, len(self.comb_layers_dims)):
            x = getattr(self, 'comb_lin_{}'.format(i))(x)
            x = F.leaky_relu(x, negative_slope=0.2)
            # if i < len(self.layer_dims) - 1:
            #     x = getattr(self, 'comb_bn_{}'.format(i))(x)
            # x = self.drop(x)
        return x

    def computer(self):
        return self.embedding_user.weight, self.embedding_item.weight

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        for i in range(1, len(self.layer_dims)):
            mat1 = getattr(self, 'lin_{}'.format(i)).weight
            reg_loss += (1 / 2) * (mat1.norm(2).pow(2) / mat1.shape[0])
        # pos_scores = self.helper(torch.cat([users_emb, pos_emb], dim=1))
        # neg_scores = self.helper(torch.cat([users_emb, neg_emb], dim=1))
        pos_scores = self.helper((users_emb, pos_emb))
        neg_scores = self.helper((users_emb, neg_emb))
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss


class CMN(BasicModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(CMN, self).__init__()
        self.dataset = dataset
        self.config = config
        self.num_sample = 10
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.padding_num = -2 ** 32 + 1
        self.embedding_user = nn.Parameter(torch.rand(size=(self.num_users, self.latent_dim), requires_grad=True),
                                           requires_grad=True)
        self.embedding_item = nn.Parameter(torch.rand(size=(self.num_items, self.latent_dim), requires_grad=True),
                                           requires_grad=True)
        #     nn.Embedding(
        #     num_embeddings=self.num_users,embedding_dim=self.latent_dim
        # )
        # self.embedding_item=nn.Embedding(
        #     num_embeddings=self.num_items,embedding_dim=self.latent_dim
        # )
        nn.init.normal_(self.embedding_item, std=0.1)
        nn.init.normal_(self.embedding_user, std=0.1)
        # 获得用户矩阵,0,1矩阵
        self.W = nn.Linear(self.latent_dim, self.latent_dim)
        self.U = nn.Linear(self.latent_dim, self.latent_dim)
        self.b = nn.Parameter(torch.randn(1, self.latent_dim), requires_grad=True)
        self.v = nn.Parameter(torch.randn(self.latent_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.b)
        print('CMN is ready to go')
        self.sampled_user = self.dataset.getUserGraph(dense=True).argsort(dim=1)[:, -self.num_sample:].to(
            world_config['device'])
        print('sample finish')

    def getUsersRating(self, users, all_users=None, all_items=None, sigmod=True):
        # def getUsersRating(self, users, all_users=None, all_items=None, sigmod=True):
        #  user_emb = self.embedding_user[users.long()]
        #  item_emb = self.embedding_item
        # # rating = F.leaky_relu(torch.mm(self.v * user_emb, item_emb.t()), negative_slope=0.2)
        #  rating=self.v*F.leaky_relu(torch.mul(user_emb,item_emb))
        #  if sigmod:
        #      return torch.sigmoid(rating)
        #  return rating
        num_batch = users.shape[0] * 10
        items = torch.arange(self.num_items)
        ratings = []
        for i in range(self.num_items // num_batch + 1):
            if i * num_batch >= self.num_items:
                break
            batch_items = items[i * num_batch: min((i + 1) * num_batch, self.num_items)]
            # print(batch_items.shape[0])
            ratings.append(self.testBatchRating(users, batch_items, sigmod=sigmod))
        rating = torch.cat(ratings, dim=1)
        # print(rating.shape)
        del ratings
        # print(rating.shape)
        return rating
        # batch_user = users.shape[0]
        # users = users.reshape(batch_user, 1).repeat(1, self.num_items).reshape(batch_user * self.num_items)
        # items = torch.arange(self.num_items).repeat(self.num_users)
        # if sigmod:
        #     output = torch.sigmoid(self.helper(users, items))
        # else:
        #     output = self.helper(users, items)
        # output = output.reshape(batch_user, self.num_items)
        # return output

    def testBatchRating(self, batch_users, batch_items, sigmod=True):
        num_user = batch_users.shape[0]
        num_item = batch_items.shape[0]
        users = batch_users.reshape(num_user, 1).repeat(1, num_item).reshape(num_user * num_item)
        items = batch_items.repeat(num_user)
        if sigmod:
            output = torch.sigmoid(self.helper(users, items))
        else:
            output = self.helper(users, items)
        output = output.reshape(num_user, num_item)
        return output

    def helper(self, users, items):
        # batch_user = self.user_mat[users.long()].to(world_config['device']) # (batch_size,num_user)
        # batch_user = torch.where(batch_user > 0, 1, 0)
        # batch_size = batch_user.shape[0]
        # adj_user_batch = batch_user.unsqueeze(-1).repeat(1, 1,
        #                                                  self.latent_dim) * self.embedding_user  # (batch_size,num_user,latent_dim)
        # M = self.embedding_user[users]  # (batch_size,latent_dim)
        # self_user_batch = M.unsqueeze(1).repeat(1, self.num_users, 1)  # (batch_size,num_user,latent_dim)
        # I = self.embedding_item[items]
        # self_item_batch = I.unsqueeze(1).repeat(1, self.num_users, 1)  # (batch_size,num_user,latent_dim)
        # # q=adj_user_batch*self_user_batch+self_item_batch*adj_user_batch
        # q = torch.bmm(self_user_batch, adj_user_batch.permute(0, 2, 1)) + torch.bmm(self_item_batch,
        #                                                                             adj_user_batch.permute(0, 2,
        #                                                                                                    1)).sum(
        #     1)  # (batch_size,num_user)
        # paddings = torch.ones_like(q) * self.padding_num
        # q = torch.where(q != 0, q, paddings)
        # q = torch.softmax(q, dim=1)
        # o = torch.sum(torch.bmm(q.unsqueeze(1), adj_user_batch), dim=1)  # (batch_size,latent_dim)
        # R = self.v * F.leaky_relu(self.U(torch.mul(M, I)) + self.W(o) + self.b)
        # R = torch.sum(R, dim=1)
        # print(users.device)
        # adj_idx = self.dataset.getUserGraph()[users.long().cpu()].to(world_config['device']).argsort(dim=1)[:, -self.num_sample:]
        adj_idx = self.sampled_user[users]
        # print(adj_idx.device)
        adj_user_embedding = self.embedding_user[adj_idx]  # (batch_size,num_sample,embedding_size)
        M = self.embedding_user[users.long()]  # (batch_size,embedding_size)
        # print(self.embedding_user.device)
        # print(M.device)
        I = self.embedding_item[items.long()]
        # print(I.shape)
        # print(I.device)
        self_user_embedding = M.unsqueeze(1).repeat(1, self.num_sample, 1)  # (batch_size,num_sample,embedding_size)
        self_item_embedding = I.unsqueeze(1).repeat(1, self.num_sample, 1)  # (batch_size,num_sample,embedding_size)
        # (batch_size,num_sample)
        q = torch.sum(
            torch.bmm(self_user_embedding, adj_user_embedding.permute(0, 2, 1)) + torch.bmm(self_item_embedding,
                                                                                            adj_user_embedding.permute(
                                                                                                0, 2, 1)), dim=1)
        paddings = torch.ones_like(q) * self.padding_num
        q = torch.where(q != 0, q, paddings)
        q = torch.softmax(q, dim=1)
        o = torch.sum(torch.bmm(q.unsqueeze(1), adj_user_embedding), dim=1)  # (batch_size,latent_dim)
        # print(M.device)
        R = self.v * F.leaky_relu(self.U(torch.mul(M, I)) + self.W(o) + self.b, negative_slope=0.2)
        R = torch.sum(R, dim=1)  # (batch_size,)
        # print(R.shape)
        return R

    def computer(self):
        return self.embedding_user, self.embedding_item

    def bpr_loss(self, users, pos, neg):
        user_emb = self.embedding_user[users.long()]
        pos_emb = self.embedding_item[pos.long()]
        neg_emb = self.embedding_item[neg.long()]
        reg_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        pos_scores = self.helper(users, pos)
        neg_scores = self.helper(users, neg)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss, reg_loss


class MMoE(nn.Module):
    def __init__(self, hidden_dim, num_tasks, num_experts, input_dim, expert_size=1,
                 use_expert_bias=True, use_gate_bias=True, alpha=0.2):
        super(MMoE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.expert_size = expert_size
        self.alpha = alpha
        self.use_gate_bias = use_gate_bias
        self.expert_kernels = nn.Parameter(torch.FloatTensor(num_experts, input_dim, hidden_dim, ))
        self.use_expert_bias = use_expert_bias
        # std = 1.0 / math.sqrt(input_dim)
        std = 0.1
        nn.init.normal_(self.expert_kernels, std=std)
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.FloatTensor(num_experts, hidden_dim, ))
            nn.init.uniform_(self.expert_bias.data, -std, std)
        # self.norm = nn.BatchNorm1d(input_dim)
        for i in range(1, expert_size):
            lin_expert_kernel = nn.Parameter(torch.FloatTensor(self.num_experts, self.hidden_dim, self.hidden_dim))
            setattr(self, 'lin_expert_kernel_{}'.format(i), lin_expert_kernel)
            nn.init.normal_(lin_expert_kernel, std=std)
            if use_expert_bias:
                lin_expert_bias = nn.Parameter(torch.FloatTensor(self.num_experts, self.hidden_dim))
                setattr(self, 'lin_expert_bias_{}'.format(i), lin_expert_bias)
                nn.init.uniform_(lin_expert_bias.data, -std, std)
            #    self.lin_expert_biases.append(lin_expert_bias)

        for i in range(1, self.num_tasks + 1):
            kernel = nn.Parameter(torch.FloatTensor(input_dim, num_experts))
            nn.init.normal_(kernel, std=std)
            setattr(self, 'gate_kernel_{}'.format(i), kernel)
            # self.gate_kernels.append(kernel)
            if use_gate_bias:
                bias = nn.Parameter(torch.FloatTensor(num_experts, ))
                nn.init.uniform_(bias, -std, std)
                setattr(self, 'gate_bias_{}'.format(i), bias)
            # self.gate_bias.append(bias)

    def forward(self, input):
        assert input.shape[-1] == self.input_dim  # batch_size*input_dim
        # expert_output = torch.mm(input, self.expert_kernels)
        expert_output = torch.bmm(input.unsqueeze(0).repeat(self.num_experts, 1, 1), self.expert_kernels).permute(1, 0,
                                                                                                                  2)
        if self.use_expert_bias:
            expert_output = torch.add(expert_output, self.expert_bias)
        # expert_output = self.norm(expert_output)
        expert_output = F.leaky_relu(expert_output, self.alpha)  # batch_size*num_expert*hidden_dim
        for i in range(1, self.expert_size):
            lin_expert_kernel = getattr(self, 'lin_expert_kernel_{}'.format(i))  # (num_expert,hidden_dim,hidden_dim)
            expert_output = torch.bmm(expert_output.permute(1, 0, 2), lin_expert_kernel).permute(1, 0, 2)
            if self.use_expert_bias:
                lin_expert_bias = getattr(self, 'lin_expert_bias_{}'.format(i))  # (num_expert,hidden_dim)
                expert_output = torch.add(expert_output, lin_expert_bias)
            # expert_output=self.norm(expert_output)
            expert_output = F.leaky_relu(expert_output, self.alpha)

        gate_outputs = []
        final_outputs = []
        for i in range(1, self.num_tasks + 1):
            gate_kernel = getattr(self, 'gate_kernel_{}'.format(i))  # input_dim*num_expert
            gate_output = torch.mm(input, gate_kernel)  # batch_size*num_expert
            if self.use_gate_bias:
                gate_output = torch.add(gate_output, getattr(self, 'gate_bias_{}'.format(i)))
            # 在softmax之前就不要加leaky relu了
            # gate_output = F.leaky_relu(gate_output, negative_slope=self.alpha)
            gate_output = F.softmax(gate_output, dim=1)  # batch_size*num_experts
            # print(gate_output[:10])
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expended_gate_output = torch.bmm(gate_output.unsqueeze(1), expert_output).squeeze(1)
            # expended_gate_output = F.leaky_relu(expended_gate_output, negative_slope=self.alpha)
            final_outputs.append(expended_gate_output)
        return final_outputs


class Mlp(nn.Module):
    def __init__(self, hidden_dim, num_tasks, input_dim, hidden_size=1,
                 use_bias=True, alpha=0.2):
        super(Mlp, self).__init__()
        self.num_tasks = num_tasks
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.alpha = alpha
        self.norm = nn.BatchNorm1d(self.hidden_dim)
        for i in range(1, self.num_tasks + 1):
            setattr(self, 'task_{}_lin_{}'.format(i, 0), nn.Linear(self.input_dim, self.hidden_dim, bias=self.use_bias))
            nn.init.xavier_uniform_(getattr(self, 'task_{}_lin_{}'.format(i, 0)).weight, gain=1)
            for j in range(1, self.hidden_size):
                setattr(self, 'task_{}_lin_{}'.format(i, j),
                        nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))
                nn.init.xavier_uniform_(getattr(self, 'task_{}_lin_{}'.format(i, j)).weight, gain=1)

    def forward(self, x):
        outputs = []
        for i in range(1, self.num_tasks + 1):
            task_x = x
            for j in range(self.hidden_size):
                task_x = getattr(self, 'task_{}_lin_{}'.format(i, j))(task_x)
                # task_x=F.leaky_relu(task_x)
                task_x = F.leaky_relu(task_x, negative_slope=0.5)
            #    task_x=self.norm(task_x)
            outputs.append(task_x)
        return outputs


class CF_MO(PairWiseModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        """
        定义CLAGL模型
        该模型为跨层聚合模型，通过跨层聚合能够更有效地捕捉协同信号
        :param config:
        :param dataset:
        """
        super(CF_MO, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.A_split = self.config['A_split']
        self.keep_prob = self.config['keep_prob']
        self.num_experts = self.config['num_experts']
        self.leaky_alpha = self.config['leaky_alpha']
        self.reg_alpha = self.config['reg_alpha']  # 回归的部分次方
        self.w1 = self.config['w1']
        self.w2 = self.config['w2']
        self.attn_weight = self.config['attn_weight']
        self.loss_mode = self.config['loss_mode']
        self.neighbor_num = self.config['neighbor_num']
        self.dropout = self.config['dropout']
        self.keep_prob = self.config['keep_prob']
        self.train_mul = int(self.config['multi_action'])
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        # stdv = 0.5 / math.sqrt(self.latent_dim)
        self.stdv = 0.1
        nn.init.normal_(self.embedding_user.weight, std=self.stdv)
        nn.init.normal_(self.embedding_item.weight, std=self.stdv)

        if self.attn_weight:
            w_u = torch.zeros(self.num_users, self.n_layers + 1, )
            w_i = torch.zeros(self.num_items, self.n_layers + 1, )
            nn.init.normal_(w_u, std=0.01)
            nn.init.normal_(w_i, std=0.01)
            w_u = w_u.to(world_config['device'])
            w_i = w_i.to(world_config['device'])
            self.w_u = nn.Parameter(w_u)
            self.w_i = nn.Parameter(w_i)

    def __dropout_x(self, x, keep_prob):
        """
        设计两种dropout方法，第一种为边dropout,也就是默认dropout方法，第二种为节点dropout,也就是随机删除节点。
        :param x:
        :param keep_prob:
        :return:
        """
        # x是一个稀疏图
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, graph):
        if self.A_split:
            graphs = []
            for g in graph:
                graphs.append(self.__dropout_x(g, keep_prob))
        else:
            graphs = self.__dropout_x(graph, keep_prob)
        return graphs

    def getColdStartEmbedding(self, users):
        users_emb = torch.matmul(users, self.embedding_item.weight)
        return users_emb

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class MultiActionModel(CF_MO):
    def __init__(self, config: dict, dataset: BasicDataset):
        """
        定义CLAGL模型
        该模型为跨层聚合模型，通过跨层聚合能够更有效地捕捉协同信号
        :param config:
        :param dataset:
        """
        super(MultiActionModel, self).__init__(config, dataset)
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.train_mul = int(self.config['multi_action'])
        self.multi_action_type = self.config['multi_action_type']
        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)

        print('graph has already loaded')
        for i in range(self.n_layers):
            setattr(self, 'layer_{}'.format(i + 1),
                    LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False, dropout=False,
                                   keep_prob=self.keep_prob)
                    )
        if self.multi_action_type == 'mmoe':
            self.user_transform = MMoE(hidden_dim=self.latent_dim, num_tasks=2, num_experts=self.num_experts,
                                       input_dim=self.latent_dim, expert_size=4, alpha=self.leaky_alpha,
                                       use_expert_bias=True, use_gate_bias=True)
            self.item_transform = MMoE(hidden_dim=self.latent_dim, num_tasks=2, num_experts=self.num_experts,
                                       input_dim=self.latent_dim, expert_size=4, alpha=self.leaky_alpha,
                                       use_expert_bias=True, use_gate_bias=True)
        else:
            self.user_transform = Mlp(hidden_dim=self.latent_dim, num_tasks=1, input_dim=self.latent_dim, hidden_size=4,
                                      use_bias=True)
            self.item_transform = Mlp(hidden_dim=self.latent_dim, num_tasks=1, input_dim=self.latent_dim, hidden_size=4,
                                      use_bias=True)

    def computer(self):
        """
        获得用户和物品的嵌入向量
        :return:
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        # all_emb = torch.cat([users_emb, items_emb], dim=0)

        users = [users_emb]
        items = [items_emb]

        for i in range(self.n_layers):
            users_emb, items_emb = getattr(self, 'layer_{}'.format(i + 1))([users_emb, items_emb])
            users.append(users_emb)
            items.append(items_emb)

        if self.attn_weight:
            user_cat_emb = torch.cat(users, dim=1).reshape(num_user, -1,
                                                           self.latent_dim)  # （num_user,n_layers,embedding_dim）
            items_cat_emb = torch.cat(items, dim=1).reshape(num_item, -1, self.latent_dim)
            weights = torch.softmax(self.w_u, dim=1)
            users_emb = torch.bmm(weights.unsqueeze(1), user_cat_emb).squeeze(1)
            weights = torch.softmax(self.w_i, dim=1)
            items_emb = torch.bmm(weights.softmax(1).unsqueeze(1), items_cat_emb).squeeze(1)
        else:
            users_emb = torch.stack(users, dim=1)  # (batch_size,n_layers,embedding_dim)
            users_emb = torch.mean(users_emb, dim=1)
            items_emb = torch.stack(items, dim=1)
            items_emb = torch.mean(items_emb, dim=1)
        return users_emb, items_emb

    def getUsersRating(self, users):
        # 这里的all_users和all_items为computer的结果
        # 这里面会加入MMoE模块
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        if not self.train_mul:
            ratings = torch.matmul(users_emb, items_emb.t())
        else:
            user_emb_list = self.user_transform(users_emb)
            item_emb_list = self.item_transform(items_emb)
            if self.loss_mode == 'mse':
                user_emb_mse = user_emb_list[0]
                item_emb_mse = item_emb_list[0]
                if self.multi_action_type == 'mmoe':
                    ratings = torch.multiply(torch.mm(user_emb_list[0], item_emb_list[0].t()),
                                             torch.pow(torch.abs(torch.mm(user_emb_list[1], item_emb_list[1].t())),
                                                       self.reg_alpha))
                else:
                    ratings = torch.multiply(torch.mm(users_emb, items_emb.t()),
                                             torch.pow(torch.abs(torch.mm(user_emb_mse, item_emb_mse.t())),
                                                       self.reg_alpha))
            else:
                ratings = torch.multiply(torch.mm(users_emb, items_emb.t()),
                                         torch.pow(torch.mm(user_emb_list[0], item_emb_list[0].t()),
                                                   self.reg_alpha))  # 这里加pow效果不佳
        return ratings

    def loss(self, users, pos, neg, score):
        users_emb, pos_emb, neg_emb, users_emb0, pos_emb0, neg_emb0 = self.getEmbedding(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)
                              ) / float(len(users))

        user_emb_list = self.user_transform(users_emb)
        pos_item_emb_list = self.item_transform(pos_emb)
        neg_item_emb_list = self.item_transform(neg_emb)
        if not self.train_mul:
            pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # (num_user,) 可以考虑在这里加入权重
            neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)  # (num_user,)
            score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
            return self.w1 * score1, 0 * score1, reg_loss
        else:
            user_emb_mse=user_emb_list[0]
            pos_item_emb_mse=pos_item_emb_list[0]
            neg_item_emb_mse=neg_item_emb_list[0]
            if self.multi_action_type == 'mmoe':
                pos_scores1 = torch.sum(torch.mul(user_emb_list[0], pos_item_emb_list[0]), dim=1)
                neg_scores1 = torch.sum(torch.mul(user_emb_list[0], neg_item_emb_list[0]), dim=1)
                pred_pos = torch.abs(torch.sum(torch.mul(user_emb_list[1], pos_item_emb_list[1]), dim=1))
                pred_neg = torch.abs(torch.sum(torch.mul(user_emb_list[1], neg_item_emb_list[1]), dim=1))
            else:
                pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # (num_user,) 可以考虑在这里加入权重
                neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)  # (num_user,)
                pred_pos = torch.abs(torch.sum(torch.mul(user_emb_mse, pos_item_emb_mse), dim=1))
                pred_neg = torch.abs(torch.sum(torch.mul(user_emb_mse, neg_item_emb_mse), dim=1))
                # print('pred_pos:{}'.format(pred_pos))
                # print('pred_neg:{}'.format(pred_neg))
            score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
            neg_score = torch.zeros_like(score, device=score.device)
            score2 = F.mse_loss(pred_pos, score) + F.mse_loss(pred_neg, neg_score)
            return self.w1 * score1, self.w2 * score2, reg_loss


# 主模型
class CLAGL(CF_MO):
    def __init__(self, config: dict, dataset: BasicDataset):
        """
        定义CLAGL模型
        该模型为跨层聚合模型，通过跨层聚合能够更有效地捕捉协同信号
        :param config:
        :param dataset:
        """
        super(CLAGL, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        # 这里还没有归一化
        self.user_graph = self.dataset.getUserGraph(dense=False)
        print('user mat loaded')
        self.item_graph = self.dataset.getItemGraph(dense=False)
        print('item mat loaded')
        if not config['top_k_graph']:
            self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        """
        这里可以选择构建topK的图
        """
        if config['top_k_graph']:
            self.graph = self.dataset.getTopKGraph()
            print('get topK graph')
        if self.n_layers > 2 and self.config['third_graph']:
            # 在这里选择已经构建好的三阶图来进行GNN传播
            self.third_graph = self.dataset.getThirdGraph()
        print('graph has already loaded')
        for i in range(self.n_layers):
            setattr(self, 'layer_{}'.format(i + 1),
                    LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False, dropout=False,
                                   keep_prob=self.keep_prob)
                    )
        # self.pna_layer = PNA_layer(self.latent_dim, self.graph, self.unit_graph, self.diag_graph, activation=False,
        #                            non_linear=False)
        self.user_item_layer = Light_UI_layer(latent_dim=self.latent_dim, user_mat=self.user_graph,
                                              item_mat=self.item_graph,
                                              activation=False, non_linear=False, dropout=False,
                                              keep_prob=self.keep_prob)
        self.light_gcn_layer = LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False,
                                              dropout=self.dropout, keep_prob=self.keep_prob)
        self.clagl_layer = CLAGL_layer(graph=self.graph, user_graph=self.user_graph, item_graph=self.item_graph)

    def computer(self):
        """
        获得用户和物品的嵌入向量
        :return:
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        # all_emb = torch.cat([users_emb, items_emb], dim=0)

        users = [users_emb]
        items = [items_emb]

        graph = self.graph
        user_graph = self.user_graph
        item_graph = self.item_graph

        # users_emb1, items_emb1, users_emb2, items_emb2 = self.clagl_layer(users_emb, items_emb)

        users_emb1, items_emb1 = self.light_gcn_layer([users_emb, items_emb], graph)
        users.append(users_emb1)
        items.append(items_emb1)
        users_emb2, items_emb2 = self.user_item_layer([users_emb, items_emb], [user_graph, item_graph])
        # users_emb2, items_emb2 = self.user_item_layer([users_emb, items_emb])
        users.append(users_emb2)
        items.append(items_emb2)

        # 这里给出两种构建三阶关系的方法，一种是2+1，一种是直接三阶。这里选择2+1的方法，较好
        if self.n_layers > 2:
            # 添加第三层
            if self.config['third_graph']:
                users_emb3, items_emb3 = self.light_gcn_layer([users_emb, items_emb], self.third_graph)
                users.append(users_emb3)
                items.append(items_emb3)
            else:
                users_emb3, items_emb3 = self.light_gcn_layer([users_emb2, items_emb2])
                # users_emb3, items_emb3 = self.user_item_layer([users_emb1, items_emb1])
                users.append(users_emb3)
                items.append(items_emb3)

        if self.n_layers > 3:
            # 构建第四阶计算方法，默认情况下为2+2
            users_emb4, items_emb4 = self.user_item_layer([users_emb2, items_emb2])
            users.append(users_emb4)
            items.append(items_emb4)

        if self.attn_weight:
            user_cat_emb = torch.cat(users, dim=1).reshape(num_user, -1,
                                                           self.latent_dim)  # （num_user,n_layers,embedding_dim）
            items_cat_emb = torch.cat(items, dim=1).reshape(num_item, -1, self.latent_dim)
            weights = torch.softmax(self.w_u, dim=1)
            users_emb = torch.bmm(weights.unsqueeze(1), user_cat_emb).squeeze(1)
            weights = torch.softmax(self.w_i, dim=1)
            items_emb = torch.bmm(weights.softmax(1).unsqueeze(1), items_cat_emb).squeeze(1)
        else:
            users_emb = torch.stack(users, dim=1)  # (batch_size,n_layers,embedding_dim)
            users_emb = torch.mean(users_emb, dim=1)
            items_emb = torch.stack(items, dim=1)
            items_emb = torch.mean(items_emb, dim=1)
        return users_emb, items_emb

    def getUsersRating(self, users):
        # 这里的all_users和all_items为computer的结果
        # 这里面会加入MMoE模块
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        ratings = torch.matmul(users_emb, items_emb.t())
        return ratings

    def loss(self, users, pos, neg, score):
        users_emb, pos_emb, neg_emb, users_emb0, pos_emb0, neg_emb0 = self.getEmbedding(users, pos, neg)
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)
                              ) / float(len(users))
        pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # (num_user,) 可以考虑在这里加入权重
        neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)  # (num_user,)
        score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
        return self.w1 * score1, 0 * score1, reg_loss


class CLAGL_Social(CLAGL):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(CLAGL_Social, self).__init__(config, dataset)
        self.__init_weight()

    def __init_weight(self):
        self.user_social_graph = self.dataset.getSocialGraph(dense=False)
        print('social network loaded')
        # self.user_graph = self.dataset.getUserGraph(dense=False)
        # print('user mat loaded')
        # self.item_graph = self.dataset.getItemGraph(dense=False)
        # print('item mat loaded')

        # self.light_gcn_layer = LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False,
        #                                       dropout=self.dropout, keep_prob=self.keep_prob)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        users = [users_emb]
        items = [items_emb]
        graph = self.graph
        ## layer1
        users_emb1, items_emb1 = self.light_gcn_layer([users_emb, items_emb], graph)
        users.append(users_emb1)
        items.append(items_emb1)
        ##layer2
        items_emb2 = torch.sparse.mm(self.item_graph, items_emb)
        items.append(items_emb2)

        users_emb2_1 = torch.sparse.mm(self.user_social_graph, users_emb)

        users_emb2_2 = torch.sparse.mm(self.user_graph, users_emb)
        users_emb2 = (users_emb2_1 + users_emb2_2) / 2.

        users.append(users_emb2)
        users_emb = torch.stack(users, dim=1)  # (batch_size,n_layers,embedding_dim)
        users_emb = torch.mean(users_emb, dim=1)
        items_emb = torch.stack(items, dim=1)
        items_emb = torch.mean(items_emb, dim=1)
        return users_emb, items_emb


class Gate(nn.Module):
    def __init__(self, latent_dim, mlp_dims, show=False):
        """
        mlp_dims:[128, 2]
        """
        super(Gate, self).__init__()
        self.embedding_dim = latent_dim
        self.softmax = nn.LogSoftmax(dim=1)
        self.show = show
        fc_layers = []
        for i in range(len(mlp_dims)):
            if i == 0:
                fc_layers.append(nn.Linear(latent_dim * 2, mlp_dims[i]))
            else:
                fc_layers.append(nn.Linear(mlp_dims[i - 1], mlp_dims[i]))
            if i != len(mlp_dims) - 1:
                # 加入BatchNorm以及ReLU层
                fc_layers.append(nn.BatchNorm1d(mlp_dims[i]))
                fc_layers.append(nn.ReLU(inplace=True))
                # fc_layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mlp = nn.Sequential(*fc_layers)

    def gumbel_softmax(self, logits, temperature, division_noise, hard):
        # y = F.softmax(logits, dim=1)
        y = self.gumbel_softmax_sample(logits, temperature, division_noise)  ## (0.6, 0.2, 0.1,..., 0.11)
        if hard:
            k = logits.size(1)  # k is numb of classes
            # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)  ## (1, 0, 0, ..., 0)
            y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            y = (y_hard - y).detach() + y
        else:
            # y_hard = torch.eq(y, torch.max(y, dim=1, keepdim=True)[0]).type_as(y)
            if self.show:
                print(torch.sum(y, dim=0))
            pass
        return y

    def gumbel_softmax_sample(self, logits, temperature, division_noise):
        """
        利用Gumbel分布进行采样的过程
        Draw a sample from the Gumbel-Softmax distribution
        """
        noise = self.sample_gumbel(logits)
        y = (logits + (noise / division_noise)) / temperature
        return F.softmax(y, dim=1)

    def sample_gumbel(self, logits):
        """
        Sample from Gumbel(0, 1)
        返回的大小和logits相同
        """
        from torch.autograd import Variable
        noise = torch.rand(logits.size())
        eps = 1e-20
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return Variable(noise.float()).to(world_config['device'])

    def forward(self, feature, temperature, hard,
                division_noise):  # z= batch x z_dim // #feature =  batch x num_gen x 256*8*8
        # 这里返回的相当于是一个概率的值
        time1 = time.time()
        x = self.mlp(feature)  # (batch_size,n_class) 这里的n_class就是2
        # out = F.softmax(x, dim=1)  # (batch_size,n_class) 这次采样简单点的方法，直接利用softmax
        # print(out)
        out = self.gumbel_softmax(x, temperature, division_noise, hard)  # (batch_size,n_class)
        out_value = out.unsqueeze(2)  # (batch_size,n_class,1)
        out = out_value.repeat(1, 1, self.embedding_dim)  # (batch_size,n_class,embedding_dim)
        time2 = time.time()
        # print('gate time:{}'.format(time2-time1))
        return out, torch.sum(out_value[:, 0]), torch.sum(out_value[:, 1])


class CF_SMP(nn.Module):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(CF_SMP, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.A_split = self.config['A_split']
        self.keep_prob = self.config['keep_prob']
        self.num_experts = self.config['num_experts']
        self.leaky_alpha = self.config['leaky_alpha']
        self.reg_alpha = self.config['reg_alpha']  # 回归的部分次方
        self.neighbor_num = self.config['neighbor_num']
        self.dropout = self.config['dropout']
        self.keep_prob = self.config['keep_prob']
        self.train_mul = int(self.config['multi_action'])
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )

        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        # stdv = 0.5 / math.sqrt(self.latent_dim)
        self.stdv = 0.1
        nn.init.normal_(self.embedding_user.weight, std=self.stdv)
        nn.init.normal_(self.embedding_item.weight, std=self.stdv)
        # 这里还没有归一化
        self.user_graph = self.dataset.getUserGraph(dense=False)
        print('user mat loaded')
        self.item_graph = self.dataset.getItemGraph(dense=False)
        print('item mat loaded')
        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        # self.act_fun = nn.LeakyReLU(negative_slope=0.2)
        self.act_fun = nn.ELU()
        # print(self.graph)
        """
        这里可以选择构建topK的图
        """
        print('graph has already loaded')
        self.user_item_layer = Light_UI_layer(latent_dim=self.latent_dim, user_mat=self.user_graph,
                                              item_mat=self.item_graph,
                                              activation=False, non_linear=False, dropout=False,
                                              keep_prob=self.keep_prob)
        self.light_gcn_layer = LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False,
                                              dropout=self.dropout, keep_prob=self.keep_prob)
        self.gates = []
        for i in range(self.n_layers):

            # setattr(self, 'gate_{}'.format(i), Gate(latent_dim=self.latent_dim, mlp_dims=[128, 2]))
            # self.gates.append(getattr(self, 'gate_{}'.format(i)))
            if i == 0:
                show = False
            else:
                show = False
            self.gates.append(Gate(latent_dim=self.latent_dim, mlp_dims=[128, 2], show=show).to(world_config['device']))

    def __dropout_x(self, x, keep_prob):
        """
        设计两种dropout方法，第一种为边dropout,也就是默认dropout方法，第二种为节点dropout,也就是随机删除节点。
        :param x:
        :param keep_prob:
        :return:
        """
        # x是一个稀疏图
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob, graph):
        if self.A_split:
            graphs = []
            for g in graph:
                graphs.append(self.__dropout_x(g, keep_prob))
        else:
            graphs = self.__dropout_x(graph, keep_prob)
        return graphs

    def __choosing_one(self, features, gumbel_out):
        # 如果之前计算的方法为hard的话，那么就是在linear和non-linear中选择一个，入果不是hard那么就相当于是加权求和
        feature = torch.sum(torch.mul(features, gumbel_out),
                            dim=1)  # batch x embedding_dim (or batch x embedding_dim x layer_num)
        return feature

    def computer(self, gum_temp, hard):
        """
        获得用户和物品的嵌入向量
        :return:
        """
        time1 = time.time()
        if self.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob, self.graph)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        # Init users & items embeddings
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        ## Layer 0
        all_emb_0 = torch.cat([users_emb, items_emb])
        # Residual embeddings
        embs = [all_emb_0]

        ## Layer 1
        all_emb_lin_1 = torch.sparse.mm(g_droped, all_emb_0)
        # Residual embeddings
        embs.append(all_emb_lin_1)

        ## layer 2
        users_emb2 = torch.sparse.mm(self.user_graph, users_emb)
        items_emb2 = torch.sparse.mm(self.item_graph, items_emb)

        all_emb_lin_2 = torch.cat([users_emb2, items_emb2], dim=0)
        embs.append(all_emb_lin_2)

        ## layer 3
        all_emb_lin_3 = torch.sparse.mm(g_droped, all_emb_lin_2)
        all_emb_non_3 = self.act_fun(all_emb_lin_3)  # 直接passby过来
        # Gating
        stack_embedding_3 = torch.stack([all_emb_lin_3, all_emb_non_3], dim=1)  # (num_user+num_item,2,embedding_dim)
        concat_embeddings_3 = torch.cat((all_emb_lin_3, all_emb_non_3), -1)  # (num_user+num_item,2*embedding_dim)
        gumbel_out_3, lin_count_3, non_count_3 = self.gates[0](concat_embeddings_3, gum_temp, hard,
                                                               self.config['division_noise'])
        embedding_1 = self.__choosing_one(stack_embedding_3, gumbel_out_3)
        embs.append(embedding_1)

        ## layer 4
        # all_emb_lin_4 = torch.sparse.mm(g_droped, embedding_1)
        # users_emb4 = torch.sparse.mm(self.user_graph, users_emb2)
        # items_emb4 = torch.sparse.mm(self.item_graph, items_emb2)
        # all_emb_lin_4 = torch.cat([users_emb4, items_emb4], dim=0)
        # all_emb_non_2 = self.act_fun(all_emb_lin_4)
        #
        # # Gating
        # stack_embedding_2 = torch.stack([all_emb_lin_4, all_emb_non_2], dim=1)
        # concat_embeddings_2 = torch.cat((all_emb_lin_4, all_emb_non_2), -1)
        #
        # gumbel_out_2, lin_count_4, non_count_4 = self.gates[1](concat_embeddings_2, gum_temp, hard,
        #                                                        self.config['division_noise'])
        # embedding_2 = self.__choosing_one(stack_embedding_2, gumbel_out_2)
        #
        # # Residual embeddings
        # embs.append(embedding_2)

        # Residual embeddings

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        # print('compute', time2 - time1)
        users_emb, items_emb = torch.split(light_out, [self.num_users, self.num_items])
        return users_emb, items_emb, [lin_count_3, non_count_3], embs

    def getUsersRating(self, users, gum_temp, hard):
        # 这里的all_users和all_items为computer的结果
        # 这里面会加入MMoE模块
        all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
        users_emb = all_users[users]
        items_emb = all_items
        ratings = self.act_fun(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def getEmbedding(self, users, pos_items, neg_items, gum_temp, hard):
        all_users, all_items, gating_dist, embs = self.computer(gum_temp, hard)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, gating_dist, embs

    def bpr_loss(self, users, pos, neg, gum_temp, hard):
        # print('gum_temp',gum_temp)
        # print('hard',hard)
        users_emb, pos_emb, neg_emb, users_emb0, pos_emb0, neg_emb0, gating_dist, embs = self.getEmbedding(users, pos,
                                                                                                           neg,
                                                                                                           gum_temp,
                                                                                                           hard)
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)
                              ) / float(len(users))
        pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # (num_user,) 可以考虑在这里加入权重
        neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)  # (num_user,)
        score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
        return score1, reg_loss, gating_dist, embs

    def forward(self, users, items):
        pass


class Layer(nn.Module):
    def __init__(self, latent_dim, user_mat, item_mat, graph, mode=0, alpha=0.2):
        super(Layer, self).__init__()
        self.latent_dim = latent_dim
        self.user_mat = user_mat
        self.item_mat = item_mat
        self.graph = graph
        self.alpha = alpha
        self.mode = mode
        self.norm = nn.BatchNorm1d(latent_dim)
        self.W_u = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_i = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_1 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_2 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_u, gain=1)
        nn.init.xavier_uniform_(self.W_i, gain=1)
        nn.init.xavier_uniform_(self.W_1, gain=1)
        nn.init.xavier_uniform_(self.W_2, gain=1)
        # self.W_uc = nn.Linear(2 * latent_dim, latent_dim)
        # self.W_ic = nn.Linear(2 * latent_dim, latent_dim)
        self.W_uc = nn.Parameter(torch.FloatTensor(3 * latent_dim, latent_dim))
        self.W_ic = nn.Parameter(torch.FloatTensor(3 * latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_uc, gain=1)
        nn.init.xavier_uniform_(self.W_ic, gain=1)

    def forward(self, input):
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        h_u1 = torch.sparse.mm(self.user_mat, users_emb)
        h_i1 = torch.sparse.mm(self.item_mat, items_emb)
        # h_i1 = torch.mm(h_i1, self.W_i)
        # h_i1 = self.norm(h_i1)
        # h_i1 = F.leaky_relu(h_i1, negative_slope=self.alpha)
        # users_emb, items_emb = h_u1, h_i1

        # 目前来说，除去最简单的LightGCN和加上内积的LightGCN，这个方法最好用
        # all_emb = torch.cat([users_emb, items_emb], dim=0)
        # all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        # all_emb_tran = self.norm(all_emb_tran)
        # all_emb_tran = F.leaky_relu(all_emb_tran, negative_slope=0.2)
        # users_emb, items_emb = torch.split(all_emb_tran, [num_user, num_item])
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        # all_emb = torch.mm((torch.mul(all_emb, all_emb_tran)), self.W_1) + torch.mm(all_emb_tran, self.W_2)
        # all_emb = torch.mul(all_emb_tran, all_emb) + all_emb_tran
        all_emb = all_emb_tran
        # all_emb = self.norm(all_emb)
        # all_emb = F.leaky_relu(all_emb, negative_slope=self.alpha)
        h_u2, h_i2 = torch.split(all_emb, [num_user, num_item])
        if self.mode == 0:
            return h_u1, h_i1
        else:
            return h_u2, h_i2
        # users_emb = self.norm(torch.add(h_u1, h_u2))
        # items_emb = self.norm(torch.add(h_i1, h_i2))
        # users_emb = torch.mm(torch.cat([h_u1, h_u2, torch.mul(h_u1, h_u2)], dim=1), self.W_uc)
        # users_emb = self.norm(users_emb)
        # users_emb = F.leaky_relu(users_emb, negative_slope=self.alpha)
        #
        # items_emb = torch.mm(torch.cat([h_i1, h_i2, torch.mul(h_i1, h_i2)], dim=1), self.W_ic)
        # items_emb = self.norm(items_emb)
        # items_emb = F.leaky_relu(items_emb, negative_slope=self.alpha)
        # users_emb = h_u1
        # items_emb=h_i1
        # return users_emb, items_emb


class LightGCN_layer(nn.Module):
    """
    LightGCN 层实现方法 默认不采用非线性变换以及激活函数
    """

    def __init__(self, latent_dim, graph=None, activation=False, non_linear=False, dropout=False, keep_prob=0.5):
        super(LightGCN_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.A_split = False
        self.activation = activation
        self.non_linear = non_linear
        self.dropout = dropout
        self.keep_prob = keep_prob
        if self.non_linear:
            self.w = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim))
            self.norm = nn.BatchNorm1d(latent_dim)
            # nn.init.normal_(self.w.data, std=0.1)
            nn.init.xavier_uniform_(self.w.data, gain=1)

    def __dropout_x(self, x, keep_prob):
        """
        设计两种dropout方法，第一种为边dropout,也就是默认dropout方法，第二种为节点dropout,也就是随机删除节点。
        :param x:
        :param keep_prob:
        :return:
        """
        # x是一个稀疏图
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    def __dropout(self, keep_prob, graph):
        if self.A_split:
            graphs = []
            for g in graph:
                graphs.append(self.__dropout_x(g, keep_prob))
        else:
            graphs = self.__dropout_x(graph, keep_prob)
        return graphs

    def forward(self, input, graph=None):
        if graph is None:
            graph = self.graph
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        if self.dropout and self.training:
            # 在每一层中进行dropout，避免第一层和第三层结果相同
            graph = self.__dropout(keep_prob=self.keep_prob, graph=graph)
        all_emb = torch.sparse.mm(graph, all_emb)
        if self.non_linear:
            all_emb = torch.mm(all_emb, self.w)
        if self.activation:
            # all_emb = self.norm(all_emb)
            all_emb = F.leaky_relu(all_emb, negative_slope=0.2)
        h_u2, h_i2 = torch.split(all_emb, [num_user, num_item])
        return h_u2, h_i2


class CLAGL_layer(nn.Module):
    def __init__(self, graph, user_graph, item_graph):
        super(CLAGL_layer, self).__init__()
        self.graph = graph
        self.user_graph = user_graph
        self.item_graph = item_graph

    def forward(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb])
        all_emb1, user_emb2, item_emb2 = torch.sparse.mm(self.graph, all_emb), torch.sparse.mm(self.user_graph,
                                                                                               user_emb), torch.sparse.mm(
            self.item_graph, item_emb)
        user_emb1, item_emb1 = torch.split(all_emb1, [user_emb.shape[0], item_emb.shape[0]])
        return user_emb1, item_emb1, user_emb2, item_emb2


class Light_cross_layer(nn.Module):
    def __init__(self, latent_dim, graph, activation=False, non_linear=False):
        super(Light_cross_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.activation = activation
        self.non_linear = non_linear
        self.norm = nn.BatchNorm1d(latent_dim)
        if self.non_linear:
            self.W_1 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
            self.W_2 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
            nn.init.normal_(self.W_1.data, std=0.1)
            nn.init.normal_(self.W_2.data, std=0.1)

    def forward(self, input):
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        if self.non_linear:
            all_emb = torch.mm((torch.mul(all_emb, all_emb_tran)), self.W_1) + torch.mm(all_emb_tran, self.W_2)
        else:
            all_emb = torch.mul(all_emb_tran, all_emb) + all_emb_tran
        if self.activation:
            all_emb = F.leaky_relu(all_emb, negative_slope=0.2)
        user_emb, item_emb = torch.split(all_emb, [num_user, num_item])
        return user_emb, item_emb


class Light_UI_layer(nn.Module):
    def __init__(self, latent_dim, user_mat=None, item_mat=None, activation=False, non_linear=False, dropout=False,
                 keep_prob=0.5):
        super(Light_UI_layer, self).__init__()
        self.latent_dim = latent_dim
        self.A_split = False
        self.user_mat = user_mat
        self.item_mat = item_mat
        self.activation = activation
        self.non_linear = non_linear
        self.dropout = dropout
        self.keep_prob = keep_prob
        if self.non_linear:
            self.W_u = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim), requires_grad=True)
            self.W_i = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim), requires_grad=True)
            nn.init.xavier_uniform_(self.W_u, gain=1)
            nn.init.xavier_uniform_(self.W_i, gain=1)

    def __dropout_x(self, x, keep_prob):
        """
        设计两种dropout方法，第一种为边dropout,也就是默认dropout方法，第二种为节点dropout,也就是随机删除节点。
        :param x:
        :param keep_prob:
        :return:
        """
        # x是一个稀疏图
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    def __dropout(self, keep_prob, graph):
        if self.A_split:
            graphs = []
            for g in graph:
                graphs.append(self.__dropout_x(g, keep_prob))
        else:
            graphs = self.__dropout_x(graph, keep_prob)
        return graphs

    def forward(self, input, graph=None):
        if graph is None:
            user_mat, item_mat = self.user_mat, self.item_mat
        else:
            user_mat, item_mat = graph[0], graph[1]
        if self.dropout and self.training:
            user_mat = self.__dropout(self.keep_prob, user_mat)
            item_mat = self.__dropout(self.keep_prob, item_mat)
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        h_u1 = torch.sparse.mm(user_mat, users_emb)
        h_i1 = torch.sparse.mm(item_mat, items_emb)

        if self.non_linear:
            h_u1 = torch.mm(h_u1, self.W_u)
        if self.activation:
            # h_u1 = self.norm(h_u1)
            h_u1 = F.leaky_relu(h_u1, negative_slope=0.2)

        if self.non_linear:
            h_i1 = torch.mm(h_i1, self.W_i)
        if self.activation:
            # h_i1 = self.norm(h_i1)
            h_i1 = F.leaky_relu(h_i1, negative_slope=0.2)
        return h_u1, h_i1


class PNA_layer(nn.Module):
    def __init__(self, latent_dim, graph, unit_graph, diag_graph, activation=False, non_linear=False):
        super(PNA_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.unit_graph = unit_graph
        self.diag_graph = diag_graph
        self.activation = activation
        self.non_linear = non_linear
        if self.non_linear:
            self.lin = nn.Linear(2 * self.latent_dim, latent_dim)

    def forward(self, input):
        """
        这里是PNA聚合和均值聚合的 加和，求一个均值
        :param input:
        :return:
        """
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        gcn_all_emb = torch.sparse.mm(self.graph, all_emb)
        # self_user, self_item = torch.split(self_all_emb, [num_user, num_item])
        sum_emb = torch.sparse.mm(self.unit_graph, all_emb)
        pow_sum_emb = torch.square(sum_emb)
        pow_emb = torch.square(all_emb)
        sum_pow_emb = torch.sparse.mm(self.unit_graph, pow_emb)
        pna_all_emb = 0.5 * torch.sparse.mm(self.diag_graph, pow_sum_emb - sum_pow_emb)
        # h_pna_u, h_pna_i = torch.split(pna_all_emb, [num_user, num_item])
        if self.non_linear:
            tran_all_emb = torch.cat([gcn_all_emb, pna_all_emb], dim=1)
            tran_all_emb = self.lin(tran_all_emb)
        else:
            tran_all_emb = 0.5 * (gcn_all_emb + pna_all_emb) + all_emb
            # users_emb = 0.5 * (self_user + h_pna_u)
            # items_emb = 0.5 * (self_item + h_pna_i)
        if self.activation:
            tran_all_emb = F.leaky_relu(tran_all_emb, negative_slope=0.2)
        # tran_all_emb=tran_all_emb+all_emb
        users_emb, items_emb = torch.split(tran_all_emb, [num_user, num_item])
        return users_emb, items_emb
        # return h_pna_u,h_pna_i


class DHCF(PairWiseModel):
    def __init__(self, config, dataset: BasicDataset):
        super(DHCF, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        print('begin to load DHCF graph')
        self.M1, self.M2 = self.dataset.getDHCFGraph()
        # print(self.M1)
        # print(self.M2)
        print('loaded DHCF graph')
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.latent_dim
        )
        self.non_linear = False
        for i in range(self.n_layers):
            setattr(self, 'user_{}'.format(i),
                    DHCF_layer(self.M1, self.latent_dim, self.non_linear, self.non_linear, self.non_linear))
            setattr(self, 'item_{}'.format(i),
                    DHCF_layer(self.M2, self.latent_dim, self.non_linear, self.non_linear, self.non_linear))

        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        self.light_gcn_layer = LightGCN_layer(self.latent_dim, self.graph_self, )
        self.theta = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.theta.weight, gain=1)

    def computer(self):
        user_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        users = [user_emb]
        items = [item_emb]
        # for i in range(self.n_layers):
        #     user_emb = getattr(self, 'user_{}'.format(i))(user_emb)
        #     item_emb = getattr(self, 'item_{}'.format(i))(item_emb)
        #     users.append(user_emb)
        #     items.append(item_emb)
        user_emb1, item_emb1 = self.light_gcn_layer([user_emb, item_emb])
        user_emb2 = getattr(self, 'user_0')(user_emb)
        item_emb2 = getattr(self, 'item_0')(item_emb)
        users.append(user_emb1)
        users.append(user_emb2)
        items.append(item_emb1)
        items.append(item_emb2)
        user_emb = torch.stack(users, dim=1)
        user_emb = torch.mean(user_emb, dim=1)
        item_emb = torch.stack(items, dim=1)
        item_emb = torch.mean(item_emb, dim=1)
        if self.non_linear:
            user_emb = self.theta(user_emb)
            item_emb = self.theta(item_emb)
        return user_emb, item_emb

    def getUsersRating(self, users, all_users=None, all_items=None, sigmod=True):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        ratings = torch.matmul(users_emb, items_emb.t())
        return ratings

    def loss(self, users, pos, neg, ):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # print('loss:{}'.format(loss))
        return loss, reg_loss

    def bpr_loss(self, users, pos, neg):
        return self.loss(users, pos, neg)

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class DHCF_layer(nn.Module):
    def __init__(self, M, latent_dim, res_net=True, non_linear=True, activation=True):
        super(DHCF_layer, self).__init__()
        self.M = M
        self.res_net = res_net
        self.non_linear = non_linear
        self.activation = activation
        self.param = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.param, gain=1)

    def forward(self, X):
        if self.res_net:
            X = torch.sparse.mm(self.M, X) + X
        else:
            X = torch.sparse.mm(self.M, X)
        if self.non_linear:
            X = torch.mm(X, self.param)
        if self.activation:
            X = F.leaky_relu(X, negative_slope=0.2)
        return X
