import torch

from dataloader import *


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Include graph information
    LastFM dataset
    """

    def __init__(self, path):
        super(LastFM, self).__init__()
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        # trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        self.path = path
        self.split = config['A_split']
        trainData = pd.read_table(path + '/data1.txt', header=None)
        # print(trainData.head())
        testData = pd.read_table(path + '/test1.txt', header=None)
        # print(testData.head())
        trustNet = pd.read_table(path + '/trustnetwork.txt', header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData -= 1
        testData -= 1
        # 这里分数减少1后，会把原来为1的值变为0。
        trainData[:][2] += 1
        testData[:][2] += 1
        # self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.trainScore = np.array(trainData[:][2])
        # print('trainUser:{}'.format(len(self.trainUser)))
        # print('tainScore:{}'.format(len(self.trainScore)))
        self.trainUniqueItems = np.unique(self.trainItem)
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.testScore = np.array(testData[:][2])
        # 进行归一化
        self.score = np.concatenate([self.trainScore, self.testScore], axis=0).astype(np.int32)
        # self.score = preprocessing.MinMaxScaler((1,5)).fit_transform(self.score.reshape(-1, 1)).reshape(-1, )
        self.score = utils.transform_score(self.score)
        # print(self.score)
        self.trainScore, self.testScore = np.split(self.score, [self.trainScore.shape[0]])
        self.unique_user = np.unique(np.concatenate([self.trainUniqueUsers, self.testUniqueUsers]))
        self.unique_item = np.unique(np.concatenate([self.trainItem, self.testItem]))
        self.n_users = int(np.max(self.unique_user) + 1)
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        # (users,users)
        # self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
        #                             shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        # 这个矩阵的shape为 （num_user*num_item）
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        print('number of user:{}'.format(self.n_users))
        print('number of item:{}'.format(self.m_items))
        if config['neighbor_num'] == -1:
            self.neighbor_num = int(math.log(self.m_items))  # config['neighbor_num']
        else:
            self.neighbor_num = config['neighbor_num']
        print('number of neighbor to construct user and item graph:{}'.format(self.neighbor_num))
        self.UserItemScoreNet = csr_matrix((self.trainScore, (self.trainUser, self.trainItem)),
                                           shape=(self.n_users, self.m_items))
        if config['delete_user']:
            self.delete_list = [20, 100, 200, 400, 600, 800, 1000, 1200, 1500]
            self.delete_lists = {}
            self.deleteUser()  # 进行删改
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)), flag=False)
        self._allPosScores = self.getUserPosItemsScore(list(range(self.n_users)))
        # print(self._allPosScores)
        # self._allPosScores = list(self._allPosScores)
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        print('begin to construct sum probality list')
        user_item_mat = csc_matrix(self.UserItemNet)
        # 注意这里list1长度应该和物品个数相同
        prob_list = [len(user_item_mat[:, i].nonzero()[0]) * 10 for i in range(self.m_items)]
        # print(list1)
        # print('list1 size:{}'.format(len(list1)))
        sum_val = 0
        sum_list = [0.]
        for i in range(self.m_items):
            sum_val += prob_list[i]
            sum_list.append(sum_val)
        # print(sum_list)
        self.prob_list = prob_list
        self.sum_list = sum_list
        # print(sum_list)
        # for i in range(self.n_users):
        #     list1.append(len(self.UserItemNet[:, i].nonzero()[0]))
        print('end construct')

    def deleteUser(self):
        """
        这个函数的使用作用为测试冷启动的效果如何
        将会根据给定的列表删除部分物品对应的连接，然后再进行测试。
        :return:
        """
        rows, cols = [], []
        data = []
        for i in range(self.n_users):
            if i in self.delete_list:
                self.delete_lists[i] = self.UserItemNet[i].nonzero()[1]
                continue
            cols.extend(self.UserItemNet[i].nonzero()[1])
            rows.extend([i] * len(self.UserItemNet[i].nonzero()[1]))
            data.extend(self.UserItemScoreNet[i].data)
        # 处理一阶关系图
        self.UserItemNet = csr_matrix((np.ones(len(cols)), (rows, cols)),
                                      shape=(self.n_users, self.m_items))
        self.UserItemScoreNet = csr_matrix((data, (rows, cols)), shape=(self.n_users, self.m_items))

    def getDeleteUserGraph(self, users):
        num_delete_user = len(users)
        graph = torch.zeros(num_delete_user, self.m_items, device=world_config['device'])
        for i, user in enumerate(users):
            graph[i, self.delete_lists[user]] = 1. / len(self.delete_lists[user])
        return graph

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
        return self._allPosScores

    # def getSparseGraph(self, add_self=False, csr=False):
    #     if self.Graph is None:
    #         user_dim = torch.LongTensor(self.trainUser)
    #         item_dim = torch.LongTensor(self.trainItem)
    #         first_sub = torch.stack([user_dim, item_dim + self.n_users])
    #         # print("first_sub:{}".format(first_sub.shape))
    #         second_sub = torch.stack([item_dim + self.n_users, user_dim])
    #         # print("second_sub:{}".format(second_sub.shape))
    #         index = torch.cat([first_sub, second_sub], dim=1)
    #         # print("index shape:{}".format(index.shape))
    #         data = torch.ones(index.size(-1)).float()
    #         # print("data shape:{}".format(data.shape))
    #         self.Graph = torch.sparse.FloatTensor(index, data,
    #                                               torch.Size(
    #                                                   [self.n_users + self.m_items, self.n_users + self.m_items]))
    #         row = torch.arange(0, self.n_users + self.m_items)
    #         col = torch.arange(0, self.n_users + self.m_items)
    #         index = torch.stack([row, col])
    #         I = torch.sparse.FloatTensor(index, torch.ones(self.n_users + self.m_items),
    #                                      torch.Size((self.n_users + self.m_items, self.m_items + self.n_users)))
    #         self.Graph_self = self.Graph + I
    #         self.Graph = utils.normalize_tensor_graph(self.Graph, mode=2)  # self.graph_normalization_tensor(self.Graph)
    #         # self.Graph_self = self.Graph + I  # 在这里需要更改graph_self的构造方法，graph_self需要在归一化之后，再加上自己，不将I进行归一化。
    #         # self.Graph_self = self.graph_normalization_tensor(self.Graph_self)
    #         self.Graph_self = utils.normalize_tensor_graph(self.Graph, mode=2)
    #         self.Graph = self.Graph.coalesce().to(world_config['device'])
    #         # self.Graph_self = self.graph_normalization_tensor(self.Graph_self)
    #         self.Graph_self = self.Graph_self.coalesce().to(world_config['device'])
    #         # self.top_k_graph = self.graph_helper(self.top_k_graph)
    #         # self.top_k_graph = self.top_k_graph.coalesce().to(world_config['device'])
    #     if add_self:
    #         # return self.top_k_graph, self.Graph_self
    #         return self.Graph, self.Graph_self
    #     return self.Graph
    #     # return self.top_k_graph
    def getSparseGraph(self, add_self=False):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                norm_adj = pre_adj_mat
                if add_self:
                    pre_adj_mat_self = sp.load_npz(self.path + '/s_pre_adj_mat_self.npz')
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
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                if add_self:
                    adj_mat_self = adj_mat + sp.eye(adj_mat.shape[0])
                    rowsum_self = np.array(adj_mat_self.sum(axis=1))
                    d_inv_self = np.power(rowsum_self, -0.5).flatten()
                    d_inv_self[np.isinf(d_inv_self)] = 0.
                    d_mat_self = sp.diags(d_inv_self)
                    norm_adj_self = d_mat_self.dot(adj_mat_self)
                    norm_adj_self = norm_adj_self.dot(d_mat_self)
                    norm_adj_self = norm_adj_self.tocsr()
                    sp.save_npz(self.path + '/s_pre_adj_mat_self.npz', norm_adj_self)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(world_config['device'])
            if add_self:
                self.Graph_self = self._convert_sp_mat_to_sp_tensor(norm_adj_self)
                self.Graph_self = self.Graph.coalesce().to(world_config['device'])
            print("don't split the matrix")
        if add_self:
            return self.Graph, self.Graph_self
        return self.Graph

    def getDegMask(self):
        """
        返回用户所有的度中，小于指定比例的mask
        如果小于，那么返回true
        否则返回false
        :return:
        """
        user_degs = []  # 保存用户的度
        item_degs = []  # 保存物品的度
        for i in range(self.n_users):
            user_degs.append(len(self.UserItemNet[i].nonzero()[0]))
        for i in range(self.m_items):
            item_degs.append(len(self.UserItemNet[:, i].nonzero()[0]))
        degs = user_degs[:]
        degs.sort()
        val = degs[(len(degs) // 5) * 3]

        mask_user = np.array(user_degs) > val
        degs = item_degs[:]
        degs.sort()
        val = degs[(len(degs) // 5) * 3]

        mask_item = np.array(item_degs) > val
        return mask_user, mask_item

    def load_adjacency_list_data(self):
        """
        获得用csr表示的数据以及对应的列表
        :return:
        """
        adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.UserItemNet.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        tmp = adj_mat.tocoo()
        all_h_list = list(tmp.row)
        all_t_list = list(tmp.col)
        all_v_list = list(tmp.data)
        return adj_mat, all_h_list, all_t_list, all_v_list

    def getTopKGraph(self):
        print('begin walk and find topK')
        self.top_k_graph = utils.preprocess_adjacency_graph(self.UserItemNet, self.n_users, self.m_items)
        print('finish walk and find topK')
        # self.top_k_graph = utils.preprocess_topk_score_graph(self.UserItemScoreNet, self.n_users, self.m_items)
        # self.top_k_graph = utils.preprocess_random_select_graph(self.UserItemNet, self.n_users, self.m_items)
        # self.top_k_graph = self.graph_normalization_tensor(self.top_k_graph)
        self.top_k_graph = utils.normalize_tensor_graph(self.top_k_graph, mode=0)
        self.top_k_graph = self.top_k_graph.coalesce().to(world_config['device'])
        return self.top_k_graph

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
            # print(self.UserItemNet[user].nonzero())
            if flag and config['delete_user'] and user in self.delete_lists:
                posItems.append(self.delete_lists[user])
                continue
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserPosItemsScore(self, users):
        scores = []
        # print('-------')
        for user in users:
            scores.append(self.UserItemScoreNet[user].data)
        return scores

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def getSampleSumRate(self):
        # return np.array(self.prob_list)
        return np.array(self.sum_list)

    def getSampleProbRate(self):
        return np.array(self.prob_list)

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

    def getUserGraph(self, dense=False):
        num = self.neighbor_num
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
            user_mat = self.construct_similarity_matrix(user_mat, num)
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
        num = self.neighbor_num
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
            item_mat = self.construct_similarity_matrix(item_mat, num)
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

    def getThirdGraph(self):
        num = self.neighbor_num
        third_path = "{0}{1}".format(self.path,
                                     '/third_graph_distance_measure_{}_neighbor_num_{}_{}.npz'.format(
                                         config['distance_measure'],
                                         config['adj_top_k'],
                                         num))
        user_path = "{0}{1}".format(self.path,
                                    '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(
                                        config['distance_measure'],
                                        num))
        if os.path.exists(third_path):
            norm_adj = sp.load_npz(third_path)
        else:
            if os.path.exists(user_path):
                user_mat = sp.load_npz(user_path)
            else:
                print('Building user distance matrix')
                user_mat = utils.construct_distance_matrix(self.UserItemNet, True)
                print('Building user distance matrix finished')
                print('Building user similarity matrix')
                user_mat = self.construct_similarity_matrix(user_mat, num)
                print('Building user similarity matrix finished')
                sp.save_npz(user_path, user_mat)
            print('Building third graph')
            third_graph = user_mat.dot(self.UserItemNet)
            third_graph = utils.construct_similar_third_graph(third_graph, num=config['adj_top_k'])
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = third_graph.tolil()
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
            sp.save_npz(third_path, norm_adj)
            print('Building third graph finished')
        third_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        # self.Graph = self.Graph.coalesce().to(world_config['device'])
        third_graph = third_graph.coalesce().to(world_config['device'])
        return third_graph

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
        '''
        转化为
        tensor形式
        '''
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
