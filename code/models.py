import numpy as np
import tensorflow.python.util.protobuf.compare
import torch

from model import *
from model import PairWiseModel

config = world.config
world_config = world.world_config


class CF_SSL(PairWiseModel):
    def __init__(self, config: dict, dataset: BasicDataset):
        super(CF_SSL, self).__init__()
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

        self.ssl_ratio = self.config["ssl_ratio"]  # 默认为0.5
        self.ssl_temp = self.config["ssl_temp"]  # 默认为0.5
        self.ssl_reg = self.config["ssl_reg"]  # 默认为0.5
        self.ssl_mode = self.config['ssl_mode']

        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        self.stdv = 0.1
        nn.init.normal_(self.embedding_user.weight, std=self.stdv)
        nn.init.normal_(self.embedding_item.weight, std=self.stdv)
        # 这里还没有归一化
        self.user_graph = self.dataset.getUserGraph(dense=False)
        print('user mat loaded')
        self.item_graph = self.dataset.getItemGraph(dense=False)
        print('item mat loaded')
        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        # self.third_graph = self.dataset.getThirdGraph()
        # print(self.graph)
        """
        这里可以选择构建topK的图
        """
        self.user_item_layer = UI_layer()
        self.light_gcn_layer = LightGCN_layer(self.latent_dim, self.graph, activation=False, non_linear=False,
                                              dropout=self.dropout, keep_prob=self.keep_prob)
        self.create_adj_mat()

    def create_adj_mat(self):
        # 默认aug_type为1
        self.graph_sub1 = self.__dropout(self.ssl_ratio, self.graph)
        self.graph_sub2 = self.__dropout(self.ssl_ratio, self.graph)
        self.user_graph_sub1 = self.__dropout(self.ssl_ratio, self.user_graph)
        self.user_graph_sub2 = self.__dropout(self.ssl_ratio, self.user_graph)
        self.item_graph_sub1 = self.__dropout(self.ssl_ratio, self.item_graph)
        self.item_graph_sub2 = self.__dropout(self.ssl_ratio, self.item_graph)

    def __dropout_x(self, x, keep_prob):
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

    def computer(self):
        """
        获得用户和物品的嵌入向量
        :return:
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb0 = torch.cat([users_emb, items_emb], dim=0)
        all_emb0_sub1 = all_emb0
        all_emb0_sub2 = all_emb0
        all_embs = [all_emb0]
        all_embs_sub1 = [all_emb0]
        all_embs_sub2 = [all_emb0]

        all_emb1 = torch.sparse.mm(self.graph, all_emb0)
        all_emb1_sub1 = torch.sparse.mm(self.graph_sub1, all_emb0_sub1)
        all_emb1_sub2 = torch.sparse.mm(self.graph_sub2, all_emb0_sub2)
        all_embs.append(all_emb1)
        all_embs_sub1.append(all_emb1_sub1)
        all_embs_sub2.append(all_emb1_sub2)

        all_emb2 = self.user_item_layer(all_emb0, [self.user_graph, self.item_graph])
        all_emb2_sub1 = self.user_item_layer(all_emb0_sub1, [self.user_graph_sub1, self.item_graph_sub1])
        all_emb2_sub2 = self.user_item_layer(all_emb0_sub2, [self.user_graph_sub2, self.item_graph_sub2])
        # all_emb2 = torch.sparse.mm(self.graph, all_emb1)
        # all_emb2_sub1 = torch.sparse.mm(self.graph_sub1, all_emb1_sub1)
        # all_emb2_sub2 = torch.sparse.mm(self.graph_sub2, all_emb1_sub2)

        all_embs.append(all_emb2)
        all_embs_sub1.append(all_emb2_sub1)
        all_embs_sub2.append(all_emb2_sub2)
        # users_emb3, items_emb3 = self.light_gcn_layer([users_emb2, items_emb2])
        # # users_emb3, items_emb3 = self.user_item_layer([users_emb1, items_emb1])
        # users.append(users_emb3)
        # items.append(items_emb3)
        all_embs = torch.stack(all_embs, dim=1)
        all_embs = torch.mean(all_embs, dim=1)
        all_embs_sub1 = torch.stack(all_embs_sub1, dim=1)
        all_embs_sub1 = torch.mean(all_embs_sub1, dim=1)
        all_embs_sub2 = torch.stack(all_embs_sub2, dim=1)
        all_embs_sub2 = torch.mean(all_embs_sub2, dim=1)
        users_emb, items_emb = torch.split(all_embs, [self.num_users, self.num_items])
        users_emb_sub1, items_emb_sub1 = torch.split(all_embs_sub1, [self.num_users, self.num_items])
        users_emb_sub2, items_emb_sub2 = torch.split(all_embs_sub2, [self.num_users, self.num_items])
        return users_emb, items_emb, users_emb_sub1, items_emb_sub1, users_emb_sub2, items_emb_sub2

    def getColdStartEmbedding(self, users):
        users_emb = torch.matmul(users, self.embedding_item.weight)
        return users_emb

    def getUsersRating(self, users):
        # 这里的all_users和all_items为computer的结果
        # 这里面会加入MMoE模块
        all_users, all_items, _, _, _, _ = self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        ratings = torch.matmul(users_emb, items_emb.t())
        return ratings

    def loss(self, users, pos, neg, score):
        users_emb, items_emb, users_emb_sub1, items_emb_sub1, users_emb_sub2, items_emb_sub2 = self.computer()
        users_emb0 = self.embedding_user(users)
        pos_emb0 = self.embedding_item(pos)
        neg_emb0 = self.embedding_item(neg)
        # reg损失
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)
                              ) / float(len(users))
        # BPR损失
        pos_scores1 = torch.sum(torch.mul(users_emb[users], items_emb[pos]), dim=1)  # (num_user,) 可以考虑在这里加入权重
        neg_scores1 = torch.sum(torch.mul(users_emb[users], items_emb[neg]), dim=1)  # (num_user,)
        score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
        # 计算ssl损失
        ssl_loss = self.cal_ssl_loss(users_emb_sub1, items_emb_sub1, users_emb_sub2, items_emb_sub2, users, pos)
        return self.w1 * score1, self.ssl_reg * ssl_loss, reg_loss

    def cal_ssl_loss(self, users_emb_sub1, items_emb_sub1, users_emb_sub2, items_emb_sub2, users, items):
        if config['ssl_mode'] == 'both_side':
            users_emb1 = users_emb_sub1[users]
            users_emb2 = users_emb_sub2[users]
            normalize_users_emb1 = F.normalize(users_emb1, dim=1)
            normalize_users_emb2 = F.normalize(users_emb2, dim=1)
            normalize_all_users_emb2 = F.normalize(users_emb_sub2, 1)
            pos_score_user = torch.sum(torch.multiply(normalize_users_emb1, normalize_users_emb2),
                                       dim=1)  # (batch_size,)
            ttl_score_user = torch.matmul(normalize_users_emb1, normalize_all_users_emb2.t())  # (batch_size,num_user)
            pos_score_user = torch.exp(pos_score_user / self.ssl_temp)  # (batch_size)
            ttl_score_user = torch.sum(torch.exp(ttl_score_user / self.ssl_temp), dim=1)  # (batch_size,)
            ssl_loss_user = -torch.mean(torch.log(pos_score_user / ttl_score_user))
            items_emb1 = items_emb_sub1[items]
            items_emb2 = items_emb_sub2[items]
            normalize_items_emb1 = F.normalize(items_emb1, dim=1)
            normalize_items_emb2 = F.normalize(items_emb2, dim=1)
            normalize_all_items_emb2 = F.normalize(items_emb_sub2, 1)
            pos_score_item = torch.sum(torch.multiply(normalize_items_emb1, normalize_items_emb2), dim=1)
            ttl_score_item = torch.matmul(normalize_items_emb1, normalize_all_items_emb2.t())
            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
            ttl_score_item = torch.sum(torch.exp(ttl_score_item / self.ssl_temp), dim=1)
            ssl_loss_item = -torch.mean(torch.log(pos_score_item / ttl_score_item))
            ssl_loss = ssl_loss_item + ssl_loss_user

        else:
            # merge操作
            # 这里会有unique
            users = torch.unique(users)
            items = torch.unique(items)
            users_emb1 = users_emb_sub1[users]
            users_emb2 = users_emb_sub2[users]
            items_emb1 = items_emb_sub1[items]
            items_emb2 = items_emb_sub2[items]
            emb_merge1 = torch.cat([users_emb1, items_emb1], dim=0)
            emb_merge2 = torch.cat([users_emb2, items_emb2], dim=0)
            normalize_emb_merge1 = F.normalize(emb_merge1, 1)
            normalize_emb_merge2 = F.normalize(emb_merge2, 1)
            pos_score = torch.sum(torch.multiply(normalize_emb_merge1, normalize_emb_merge2), dim=1)
            ttl_score = torch.matmul(normalize_emb_merge1, normalize_emb_merge2.t())
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_temp), dim=1)
            ssl_loss = -torch.mean(torch.log(pos_score / ttl_score))
            ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss


class UI_layer(nn.Module):
    def forward(self, input, graph=None):
        user_mat, item_mat = graph[0], graph[1]
        users_emb, items_emb = torch.split(input, [user_mat.shape[0], item_mat.shape[0]])
        h_u1 = torch.sparse.mm(user_mat, users_emb)
        h_i1 = torch.sparse.mm(item_mat, items_emb)
        return torch.cat([h_u1, h_i1], dim=0)


