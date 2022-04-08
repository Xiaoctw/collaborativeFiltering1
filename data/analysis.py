import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

dataset = 'amazon-book'
datasets=['amazon-book-init','yelp2018','gowalla']
dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/' + dataset

train_file = '{0}/train.txt'.format(dataset_path)
train_score_file = '{0}/train_score.txt'.format(dataset_path)
test_file = '{0}/test.txt'.format(dataset_path)
test_score_file = '{0}/test_score.txt'.format(dataset_path)
user2items = defaultdict(lambda: [])
item2users = defaultdict(lambda: [])
n_user, m_item = 0, 0
with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            n_user = max(n_user, uid + 1)
            for item in items:
                user2items[uid].append(item)
                item2users[item].append(uid)
            if len(items) > 0:
                m_item = max(m_item, max(items) + 1)
print('n_user:{}'.format(n_user))
print('m_item:{}'.format(m_item))

deg_users = []
min_deg = float('inf')
max_deg = -float('inf')
for i in range(n_user):
    deg_users.append(len(user2items[i]))
    max_deg = max(max_deg, len(user2items[i]))
    min_deg = min(min_deg, len(user2items[i]))
print('用户平均度:{}'.format(sum(deg_users) / len(deg_users)))
print('最大度:{}'.format(max_deg))
print('最小度:{}'.format(min_deg))
bins = [min_deg + i * (max_deg - min_deg) / 100 for i in range(101)]
plt.hist(deg_users, bins, alpha=0.8, rwidth=0.5)
plt.title('{} dataset user deg'.format(dataset))
plt.show()

min_deg = float('inf')
max_deg = -float('inf')
deg_items=[]
for i in range(m_item):
    # deg_users.append(len(user2items[i]))
    # max_deg = max(max_deg, len(user2items[i]))
    # min_deg = min(min_deg, len(user2items[i]))
    deg_items.append(len(item2users[i]))
    max_deg = max(max_deg, len(item2users[i]))
    min_deg = min(min_deg, len(item2users[i]))

print('物品平均度:{}'.format(sum(deg_items) / len(deg_items)))
print('最大度:{}'.format(max_deg))
print('最小度:{}'.format(min_deg))
bins = [min_deg + i * (max_deg - min_deg) / 100 for i in range(101)]
print('物品平均度:{}'.format(sum(deg_items) / len(deg_items)))
plt.hist(deg_users, bins, alpha=0.8, rwidth=0.5)
plt.show()