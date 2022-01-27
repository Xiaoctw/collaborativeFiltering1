import matplotlib.pyplot as plt
import sonnet.src.initializers
import tensorflow
import torch
from scipy.sparse import *
import numpy as np
from collections import Counter
import math
import networkx as nx


row = [0, 0, 0, 1, 1, 2, 2]  # 行指标
col = [0, 1, 2, 1, 2, 1, 2]  # 列指标
data = [1, 1, 1, 1, 1, 1, 4]  # 在行指标列指标下的数字
team = csr_matrix((data, (row, col)), shape=(3, 3))

# print(team.col)
# row = [0, 0, 0, 1, 1, 2, 2]  # 行指标
# col = [0, 2, 2, 1, 2, 1, 2]  # 列指标
# data = [1, 2, 1, 1, 1, 1, 4]  # 在行指标列指标下的数字

team1 = csc_matrix((data, (row, col)), shape=(3, 3))
#
# for i in range(3):
#     print(team[:,i].nonzero())
#     print('----')
# print('------')
#
# for i in range(3):
#     print(team1[:,i].nonzero())
#     print('----')

# G = nx.DiGraph()        # 无多重边有向图
# G.add_node(2)  # 添加一个节点
# G.add_nodes_from([3, 4, 5, 6, 8, 9, 10, 11, 12])  # 添加多个节点
# # G.add_cycle([1, 2, 3, 4])  # 添加环
# G.add_edge(1, 3)  # 添加一条边
# G.add_edges_from([(3, 5), (3, 6), (6, 7)])  # 添加多条边
# G.remove_node(8)  # 删除一个节点
# G.remove_nodes_from([9, 10, 11, 12])  # 删除多个节点
# print("nodes: ", G.nodes())  # 输出所有的节点
# print("edges: ", G.edges())  # 输出所有的边
# print("number_of_edges: ", G.number_of_edges())  # 边的条数，只有一条边，就是（2，3）
# print("degree: ", G.degree)  # 返回节点的度
# print("in_degree: ", G.in_degree)  # 返回节点的入度
# print("out_degree: ", G.out_degree)  # 返回节点的出度
# print("degree_histogram: ", nx.degree_histogram(G))  # 返回所有节点的分布序列
#
# nx.draw(G,with_labels=True,node_color=['red','red','green','green','blue','blue','blue'])
# plt.show()




# for i in range(1, 10):
#     print(func(i))




    
