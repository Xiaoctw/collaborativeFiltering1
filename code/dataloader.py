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
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def allPosScores(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self, add_self=False):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def load_adjacency_list_data(self):
        """
        根据DGCF的更改
        :return:
        """
        raise NotImplementedError

    def get_diag_unit_graph(self):
        raise NotImplementedError

    def getUserGraph(self, dense=False):
        pass

    def getItemGraph(self, dense=False):
        pass

    def getThirdGraph(self):
        pass

    def getTopKGraph(self):
        raise NotImplementedError

    def getDHCFGraph(self):
        pass

    def getSampleSumRate(self):
        raise NotImplementedError

    def getSampleProbRate(self):
        raise NotImplementedError

    def getDegMask(self):
        raise NotImplementedError

    def deleteUser(self):
        """
        这个函数的使用作用为测试冷启动的效果如何
        将会根据给定的列表删除部分物品对应的连接，然后再进行测试。
        :return:
        """
        raise NotImplementedError


