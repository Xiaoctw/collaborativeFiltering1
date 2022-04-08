import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from collections import defaultdict
import matplotlib.pyplot as plt

trainData = pd.read_table('./lastfm/data1.txt', header=None)
testData = pd.read_table('./lastfm/test1.txt', header=None)
trustNet = pd.read_table('./lastfm/trustnetwork.txt', header=None).to_numpy()
trustNet -= 1
trainData -= 1
testData -= 1
trainData[:][2] += 1
testData[:][2] += 1
trainUser = np.array(trainData[:][0])
trainUniqueUsers = np.unique(trainUser)
trainItem = np.array(trainData[:][1])
trainScore = np.array(trainData[:][2])

trainUniqueItems = np.unique(trainItem)
# self.trainDataSize = len(self.trainUser)
testUser = np.array(testData[:][0])
testUniqueUsers = np.unique(testUser)
testItem = np.array(testData[:][1])
testScore = np.array(testData[:][2])
# 进行归一化
unique_user = np.unique(np.concatenate([trainUniqueUsers, testUniqueUsers]))
unique_item = np.unique(np.concatenate([trainItem, testItem]))
n_user = int(np.max(unique_user) + 1)
m_item = int(np.max(unique_item) + 1)

UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)),
                         shape=(n_user, m_item))

user2items = defaultdict(lambda: [])
item2users = defaultdict(lambda: [])
cnt = 0
print('n_user:{}'.format(n_user))
print('m_item:{}'.format(m_item))
user_degs = []
for i in range(n_user):
    user2items[i] = UserItemNet[i].nonzero()[1]
    cnt += len(user2items[i])
    user_degs.append(len(user2items))
print('用户平均度:{}'.format(cnt / n_user))
cnt = 0
item_degs = []
for j in range(m_item):
    item2users[j] = UserItemNet[:, j].nonzero()[0]
    item_degs.append(len(item2users[j]))
    cnt += len(item2users[j])
print('物品平均度:{}'.format(cnt / m_item))

print(item_degs)
plt.boxplot(item_degs)
plt.title('item deg')
plt.show()
