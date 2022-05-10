import os
import numpy as np
import pandas as pd
import torch

from dataloader import *


class Deli(BasicDataset):
    def __init__(self, path):
        super(Deli, self).__init__()
        cprint("loading [Deli]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.path = path
        self.folds = config['A_n_fold']
        self.split = config['A_split']
        trainData = pd.read_csv(path + '/train_data.csv', sep=',').to_numpy()
        # print(trainData)
        testData = pd.read_csv(path + '/test_data.csv', sep=',').to_numpy()
        self.socialData = pd.read_csv(path + '/social_data.csv', sep=',').to_numpy()
        # print(trainData[:][0])
        # print(trainData)
        self.trainUser = np.array(trainData[:, 0])
        self.trainItem = np.array(trainData[:, 1])
        self.train_unique_users = np.unique(self.trainUser)
        self.train_unique_items = np.unique(self.trainItem)
        self.testUser = np.array(testData[:, 0])
        self.testItem = np.array(testData[:, 1])
        self.test_unique_users = np.unique(self.testUser)

        self.unique_user = np.unique(np.concatenate([self.train_unique_users, self.test_unique_users]))
        self.unique_item = np.unique(np.concatenate([self.trainItem, self.testItem]))

        # print(self.trainUser)
        # print(int(np.max(self.trainUser)))
        # print(int(np.max(self.testUser)))
        self.n_users = max(int(np.max(self.trainUser)), int(np.max(self.testUser))) + 1
        # self.m_items = max(int(np.max(self.trainItem)), int(np.max(self.testItem))) + 1
        print('delicious sparsity:{}'.format(
            (len(self.trainItem) + len(self.trainUser)) / self.m_items / self.n_users
        ))

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        print('number of user:{}'.format(self.n_users))
        print("number of item:{}".format(self.m_items))
        if config['neighbor_num'] == -1:
            self.neighbor_num = int(math.log(self.m_items))  # config['neighbor_num']
        else:
            self.neighbor_num = config['neighbor_num']
        self._allPos = self.getUserPosItems(list(range(self.n_users)), flag=False)
        # self._allPosScores = self.getUserPosItemsScore(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        print('begin to construct sum probality list')

        self.Graph = None

    def _split_A_hat(self, A):
        # 意思是把图给分割了，画成了很多等份进行训练
        # 对于大规模图来说很有帮助
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world_config['device']))
        return A_fold

    @property
    def n_users(self):
        return int(np.max(self.unique_user) + 1)

    @property
    def m_items(self):
        return int(np.max(self.unique_item) + 1)

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosScores(self):
        return self._allPos

    def getSparseGraph(self, add_self=False):
        """
        获得稀疏图
        :param add_self: 是否加上自循环
        :return:
        """
        adj_mat_path = self.path + '/s_pre_adj_mat.npz'
        adj_mat_self_path = self.path + '/s_pre_adj_mat_self.npz'
        if config['delete_user'] and False:
            adj_mat_path = self.path + '/s_pre_adj_mat_delete.npz'
            adj_mat_self_path = self.path + '/s_pre_adj_mat_self_delete.npz'
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(adj_mat_path)
                norm_adj = pre_adj_mat
                if add_self:
                    pre_adj_mat_self = sp.load_npz(adj_mat_self_path)
                    norm_adj_self = pre_adj_mat_self
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print("costing {:.4f} s, saved norm_mat...".format(end - s))
                sp.save_npz(adj_mat_path, norm_adj)
                if add_self:
                    adj_mat_self = adj_mat + sp.eye(adj_mat.shape[0])
                    rowsum_self = np.array(adj_mat_self.sum(axis=1))
                    d_inv_self = np.power(rowsum_self, -0.5).flatten()
                    d_inv_self[np.isinf(d_inv_self)] = 0.
                    d_mat_self = sp.diags(d_inv_self)
                    norm_adj_self = d_mat_self.dot(adj_mat_self)
                    norm_adj_self = norm_adj_self.dot(d_mat_self)
                    norm_adj_self = norm_adj_self.tocsr()
                    sp.save_npz(adj_mat_self_path, norm_adj_self)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world_config['device'])
            if add_self:
                self.Graph_self = self._convert_sp_mat_to_sp_tensor(norm_adj_self)
                self.Graph_self = self.Graph.coalesce().to(world_config['device'])
            print("don't split the matrix")
        if add_self:
            return self.Graph, self.Graph_self
        return self.Graph

    def getSocialGraph(self, dense=False):
        num_trust = self.socialData.shape[0]
        # print(np.max(self.trustNet))
        # pass
        social_path = self.path + '/social.npz'
        if os.path.exists(social_path):
            UserSocialNet = sp.load_npz(social_path)
        else:
            # print(self.socialData)
            # print(self.n_users)
            # print(np.max(self.socialData))
            UserSocialNet = csr_matrix((np.ones(num_trust), self.socialData.transpose()),
                                       shape=(self.n_users, self.n_users))
            UserSocialNet = utils.normalize_graph(UserSocialNet, mode=1)
            sp.save_npz(social_path, UserSocialNet)
        # print(UserSocialNet.sum(1))
        if dense:
            UserSocialNet = UserSocialNet.todense()
            UserSocialNet = torch.Tensor(UserSocialNet)
            return UserSocialNet
        UserSocialNetTensor = self._convert_sp_mat_to_sp_tensor(UserSocialNet)
        UserSocialNetTensor = UserSocialNetTensor.to(world_config['device'])
        return UserSocialNetTensor

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
                users:
                    shape [-1]
                items:
                    shape [-1]
                return:
                    feedback [-1]
                """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users, flag=True):
        """
        :param users:
        :param flag: 只有在测试和coldstart测试的时候才为True
        :return:
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.train_unique_users[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.train_unique_users)

    def getUserGraph(self, dense=False):
        # 用户图
        if config['neighbor_num'] == -1:
            num = int(math.log(self.n_users))
        else:
            num = self.neighbor_num
        # num = int(math.log(self.n_users))
        path = "{0}{1}".format(self.path,
                               '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(config['distance_measure'],
                                                                                          num))
        # print(utils.construct_distance_matrix(self.UserItemNet, True))
        if os.path.exists(path):
            user_mat = sp.load_npz(path)
            # print('user mat:', user_mat)
        else:
            print('Building user distance matrix')
            user_mat = utils.construct_distance_matrix(self.UserItemNet, True)
            print('Building user distance matrix finished')
            print('Building user similarity matrix')
            # user_mat = self.construct_similarity_matrix(user_mat, num)
            user_mat = utils.construct_similar_graph(user_mat, num=num)
            print('Building user similarity matrix finished')
            sp.save_npz(path, user_mat)
        if dense:
            user_mat = user_mat.todense()
            user_mat = torch.Tensor(user_mat)
            return user_mat
        else:
            user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
        # self.Graph = self.Graph.coalesce().to(world_config['device'])
        user_mat = user_mat.coalesce().to(world_config['device'])
        # print(user_mat.shape)
        return user_mat

    def getItemGraph(self, dense=False):
        # 物品图
        if config['neighbor_num'] == -1:
            num = int(math.log(self.m_items))
        else:
            num = self.neighbor_num
        # num = int(math.log(self.m_items))
        path = "{0}{1}".format(self.path,
                               '/item_mat_distance_measure_{}_neighbor_num_{}.npz'.format(config['distance_measure'],
                                                                                          num))
        if os.path.exists(path):
            item_mat = sp.load_npz(path)
        else:
            print('Building item distance matrix')
            item_mat = utils.construct_distance_matrix(self.UserItemNet, False)
            print('Building item distance matrix finished')
            print('Building item similarity matrix')
            # item_mat = self.construct_similarity_matrix(item_mat, num)
            item_mat = utils.construct_similar_graph(item_mat, num=num)
            print('Building item similarity matrix finished')
            sp.save_npz(path, item_mat)
        if dense:
            item_mat = item_mat.todense()
            item_mat = torch.Tensor(item_mat)
            return item_mat
        else:
            item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
        # self.Graph = self.Graph.coalesce().to(world_config['device'])
        item_mat = item_mat.coalesce().to(world_config['device'])
        # print(item_mat.shape)
        return item_mat

    @staticmethod
    def construct_similarity_matrix(matrix, neighbor_num):
        """
        构建相似性矩阵的过程，假设原始的用户-物品交互矩阵为M
        :param matrix: 如果构建的为用户-用户相似性矩阵，那么 matrix=MM^T,
        如果是物品-物品相似性矩阵，那么matrix=M^TM
        :param neighbor_num: 相似的邻居个数
        :return:
        """
        dense_mat = matrix.todense()
        # 将对角线元素归一化
        dense_mat[np.arange(len(dense_mat)), np.arange(len(dense_mat))] = 0
        args = np.argsort(dense_mat, axis=1)[:, -neighbor_num:]
        idxes = np.concatenate(neighbor_num * [np.arange(len(args))], axis=0)
        idxes = idxes.reshape(neighbor_num, len(args)).transpose()
        matrix = np.zeros((len(args), len(args)))
        matrix[idxes, args] = dense_mat[idxes, args]
        _dseq = matrix.sum(1)  # 按行求和后拉直
        _dseq[_dseq == 0] = 1
        _D_half0 = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
        matrix = _D_half0.dot(matrix).dot(_D_half0)  # 矩阵乘法
        matrix[np.isnan(matrix)] = 0
        index = matrix.nonzero()
        # data = item_mat[item_mat > 0]
        data = matrix[index]
        matrix = sp.coo_matrix((data, index),
                               shape=(len(args), len(args)))
        return matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        转化为
        tensor形式
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return mat

    @n_users.setter
    def n_users(self, value):
        self._n_users = value
