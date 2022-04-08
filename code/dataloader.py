import os
import math
import sklearn.preprocessing
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import sklearn
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import world
import utils
from world import cprint
from time import time

config = world.config
world_config = world.world_config

class BasicDataset(Dataset):
    """
    数据集的基类
    """
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        """用户个数"""
        raise NotImplementedError

    @property
    def m_items(self):
        """物品个数"""
        raise NotImplementedError

    @property
    def trainDataSize(self):
        """训练集大小"""
        raise NotImplementedError

    @property
    def testDict(self):
        """测试数据"""
        raise NotImplementedError

    @property
    def allPos(self):
        """返回正类样本列表"""
        raise NotImplementedError

    @property
    def allPosScores(self):
        """
        正类样本对应的分数
        部分数据集不需使用
        只有datarate需要使用
        """
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):

        raise NotImplementedError

    def getUserPosItems(self, users):
        """
        指定用户的正类样本
        """
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        对于较大数据集不必要返回所有负类样本
        需要采样
        """
        raise NotImplementedError

    def getSparseGraph(self, add_self=False):
        """
        通过torch.sparse.IntTensor构建传播矩阵，
        在NGCF论文中详细描述了这类矩阵构建方法
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def load_adjacency_list_data(self):
        """
        根据DGCF的更改，舍弃
        :return:
        """
        raise NotImplementedError

    def get_diag_unit_graph(self):
        """
        不同于getSparseGraph,该函数同样返回Matrix+Self
        用于PNA层等复杂结构的搭建
        :return: 两个矩阵，M和M+I
        """
        raise NotImplementedError

    def getUserGraph(self, dense=False):
        """
        获取用户矩阵
        (num_user,num_user)
        :param dense:是否返回稠密矩阵，默认为false
        :return:
        """
        raise NotImplementedError

    def getItemGraph(self, dense=False):
        """
        获取物品矩阵
        (num_user,num_user)
        :param dense:是否返回稠密矩阵，默认为false
        :return:
        """
        raise NotImplementedError

    def getSocialGraph(self, dense=False):
        """
        获得用户社交网络
        (num_user,num_user)
        :param dense:
        :return:
        """
        raise NotImplementedError

    def getThirdGraph(self):
        """
        获得用户物品三阶图
        利用原始用户-物品矩阵M
        M(M^TM)来构建
        :return:三阶图
        """
        pass

    def getTopKGraph(self):
        """
        获取topK关系图
        topK关系图可以有多种构建方法
        比如说通过随机游走来构建
        比如说通过计算评分的topK来实现等
        :return:
        """
        raise NotImplementedError


    def getDHCFGraph(self):
        """
        获得DHCF图，DHCF为超图结构
        """
        pass

    def getSampleSumRate(self):
        """获得出现频次列表的累加和"""
        raise NotImplementedError

    def getSampleProbRate(self):
        """获得出现频次列表"""
        raise NotImplementedError

    def getDegMask(self):
        """
        返回用户所有的度中，小于指定比例的mask
        如果小于，那么返回true
        否则返回false
        :return:
        """
        raise NotImplementedError

    def deleteUser(self):
        """
        这个函数的使用作用为测试冷启动的效果如何
        将会根据给定的列表删除部分物品对应的连接，然后再进行测试。
        :return:
        """
        raise NotImplementedError


