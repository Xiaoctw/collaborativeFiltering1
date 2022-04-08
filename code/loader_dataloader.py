from dataloader import *


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset，yelp2018数据集
    导入相关数据集
    """

    def __init__(self, path, config=config):
        # train or test
        super().__init__()
        cprint(f'loading [{path}]')
        # print(config)
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = '{0}/train.txt'.format(path)
        test_file = '{0}/test.txt'.format(path)
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.train_data_size = 0
        self.testDataSize = 0
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.train_data_size += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                if len(l) > 1:  # amazon-book有个例外
                    items = [int(i) for i in l[1:]]
                    # print(items)
                    uid = int(l[0])
                    #  print(uid)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        print('number of user:{}'.format(self.n_user))
        print('number of item:{}'.format(self.m_item))
        if config['neighbor_num'] == -1:
            self.neighbor_num = int(math.log(self.m_items))  # config['neighbor_num']
        else:
            self.neighbor_num = config['neighbor_num']
        print('number of neighbor to construct user and item graph:{}'.format(self.neighbor_num))
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world_config['dataset']} Sparsity : {(self.train_data_size + self.testDataSize) / self.n_users / self.m_items}")
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        if config['delete_user']:
            # 不同数据集去除的用户不一样
            if world_config['dataset'] == 'yelp2018':
                self.delete_list = [100, 200, 800, 1000, 2000, 4000, 8000, 10000, 12000, 15000, 18000, 20000]
            else:
                self.delete_list = [100, 1000, 2000, 4000, 10000, 12000, 15000]
            self.delete_lists = {}
            self.deleteUser()
        # self.UserItemScoreNet = csr_matrix((self.trainScores, (self.trainUser, self.trainItem)),
        #                                    shape=(self.n_user, self.m_item))
        # 计算度矩阵
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)), flag=False)
        self._allPosScores = None
        self.__testDict = self.__build_test()
        print(f"{world_config['dataset']} is ready to go")
        user_item_mat = csc_matrix(self.UserItemNet)
        # 注意这里list1长度应该和物品个数相同
        prob_list = [int(math.pow(len(user_item_mat[:, i].nonzero()[0]) * 10, 1)) for i in range(self.m_items)]
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

    def deleteUser(self):
        """
        这个函数的使用作用为测试冷启动的效果如何
        将会根据给定的列表删除部分物品对应的连接，然后再进行测试。
        :return:
        """
        rows, cols = [], []

        for i in range(self.n_users):
            if i in self.delete_list:
                self.delete_lists[i] = self.UserItemNet[i].nonzero()[1]
                continue
            cols.extend(self.UserItemNet[i].nonzero()[1])
            rows.extend([i] * len(self.UserItemNet[i].nonzero()[1]))
        # 处理一阶关系图
        self.UserItemNet = csr_matrix((np.ones(len(cols)), (rows, cols)),
                                      shape=(self.n_users, self.m_items))

    def getDeleteUserGraph(self, users):
        """
        获得被删除用户的图，利用这个图可以构造用户的嵌入向量
        :param users: 指定用户
        :return:
        """
        num_delete_user = len(users)
        graph = torch.zeros(num_delete_user, self.m_items, device=world_config['device'])
        for i, user in enumerate(users):
            graph[i, self.delete_lists[user]] = 1. / len(self.delete_lists[user])
        return graph

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.train_data_size

    @property
    def testDict(self):
        """
        获取的就是字典，键为用户，值为对应的物品
        :return:
        """
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosScores(self):
        return self._allPos

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

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        转化为
        tensor形式
        转化后可以在GPU中运行
        """
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return mat

    def getSparseGraph(self, add_self=False):
        """
        获得稀疏图，用户-物品的交互图
        :param add_self: 是否添加自连接，是的话add_self为True
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

            if self.split:
                self.Graph = self._split_A_hat(norm_adj)
                if add_self:
                    self.Graph_self = self._split_A_hat(norm_adj_self)
                print("done split matrix")
            else:
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
        如果多目标学习评分采用交叉熵会使用该函数
        :return:
        """
        user_degs = [len(self.UserItemNet[i].nonzero()[0]) for i in range(self.n_user)]
        item_degs = [len(self.UserItemNet[:, i].nonzero()[0]) for i in range(self.m_item)]
        degs = user_degs[:]
        degs.sort()
        val = degs[(len(degs) // 5) * 4]
        mask_user = np.array(user_degs) > val
        degs = item_degs[:]
        degs.sort()
        val = degs[(len(degs) // 5) * 4]
        mask_item = np.array(item_degs) > val
        return mask_user, mask_item

    def getTopKGraph(self):
        """
        这里只支持随机游走方法
        因为这个数据集没有评分信息
        :return:
        """
        print('begin walk and find topK')
        self.top_k_graph = utils.preprocess_adjacency_graph(self.UserItemNet, self.n_users, self.m_items)
        print('finish walk and find topK')
        self.top_k_graph = self.graph_helper(self.top_k_graph)
        self.top_k_graph = self.top_k_graph.coalesce().to(world_config['device'])
        return self.top_k_graph

    def graph_helper(self, graph):
        # 相当于进行计算D-1AD操作
        # 进行归一化
        dense = graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
            [self.n_users + self.m_items, self.n_users + self.m_items]))
        graph = graph.coalesce().to(world_config['device'])
        return graph

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
        posItems = []
        for user in users:
            if flag and config['delete_user'] and user in self.delete_lists:
                posItems.append(self.delete_lists[user])
                continue
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserGraph(self, dense=False):
        #获得用户图
        user_mat_path = "{0}{1}".format(self.path,
                                        '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(
                                            config['distance_measure'],
                                            self.neighbor_num))
        if os.path.exists(user_mat_path):
            user_mat = sp.load_npz(user_mat_path)
        else:
            print('Building user distance matrix')
            user_mat = utils.construct_distance_matrix(self.UserItemNet, True)
            print('Building user distance matrix finished')
            print('Building user similarity matrix')
            user_mat = utils.construct_similar_graph(user_mat, num=self.neighbor_num)
            print('Building user similarity matrix finished')
            sp.save_npz(user_mat_path, user_mat)
        if dense:
            user_mat = user_mat.todense()
            user_mat = torch.Tensor(user_mat)
            return user_mat
        else:
            user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
        user_mat = user_mat.coalesce().to(world_config['device'])
        return user_mat

    def getItemGraph(self, dense=False):
        #获得物品图
        item_mat_path = "{0}{1}".format(self.path,
                                        '/item_mat_distance_measure_{}_neighbor_num_{}.npz'.format(
                                            config['distance_measure'],
                                            self.neighbor_num))
        if os.path.exists(item_mat_path):
            item_mat = sp.load_npz(item_mat_path)
        else:
            print('Building item distance matrix')
            item_mat = utils.construct_distance_matrix(self.UserItemNet, False)
            print('Building item distance matrix finished')
            print('Building item similarity matrix')
            item_mat = utils.construct_similar_graph(item_mat, num=self.neighbor_num)
            print('Building item similarity matrix finished')
            sp.save_npz(item_mat_path, item_mat)
        if dense:
            item_mat = item_mat.todense()
            item_mat = torch.Tensor(item_mat)
            return item_mat
        else:
            item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
        item_mat = item_mat.coalesce().to(world_config['device'])
        return item_mat

    def getThirdGraph(self):
        #获得三阶关系图
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
                user_mat = utils.construct_similar_graph(user_mat, num)
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

    def getSampleSumRate(self):
        return np.array(self.sum_list)

    def getSampleProbRate(self):
        return np.array(self.prob_list)
