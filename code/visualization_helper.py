import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import scipy.sparse as sp
# tsne = TSNE(n_components=2, init='random')
#   Y = tsne.fit_transform(X)    #后面两个参数分别是邻居数量以及投影的维度
#   pylab.scatter(Y[:, 0], Y[:, 1], 20, labels_data)
import world
import utils
import random
import model

random.seed(2021)
import numpy as np

np.random.seed(2021)
world_config = world.world_config
world_config['comment'] = 'cf_mo'
world_config['model_name'] = 'cf_mo'
config = world.config
from register import dataset

"""
visulization的辅助性工作
在这里寻找指定用户相近的K个物品
"""

user_mat_path = "{0}{1}".format(dataset.path,
                                '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(
                                    config['distance_measure'],
                                    dataset.neighbor_num))
user_mat = sp.load_npz(user_mat_path)
user_graph = user_mat.tocsr()
user_item = dataset.UserItemNet

choosed_user = [100, 200, 300]
# print(sum(user_graph[100].data))
# print(sum(user_graph[200].data))
# print(sum(user_graph[300].data))
# print(user_graph[choosed_user].data)
for user in choosed_user:
    print(user_graph[user].nonzero()[1])
# print(user_item[choosed_user])

"""
lastfm
choosed_user = [100, 200, 300]
adj_user = [74, 468, 855, 863, 1007, 1167, 1406, 1827, 23, 67, 108, 354, 426, 555, 667, 1069, 151, 368, 514
    , 736, 1071, 1678, 1781, 1850]
"""

"""
gowalla
#choosed_user = [100, 200, 300]
# adj_user = [26644 25801 22780 19546 19336 19164 17847   696 24592 17497 17340 13095
#  12656 11293  8820   205 11546 11198 10160  9507  8753  6445  4196   866]
"""

