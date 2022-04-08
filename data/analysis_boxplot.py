import os
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

dataset = 'amazon-book'
datasets=['amazon-book-init','yelp2018','gowalla']

data_user=[]
data_item=[]
for dataset in datasets:
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
    print('{}:n_user:{}'.format(dataset,n_user))
    print('{}:m_item:{}'.format(dataset,m_item))
    deg_users = []
    deg_items=[]
    min_deg = float('inf')
    max_deg = -float('inf')
    for i in range(n_user):
        deg_users.append(len(user2items[i]))
        max_deg = max(max_deg, len(user2items[i]))
        min_deg = min(min_deg, len(user2items[i]))
    for i in range(m_item):
        # deg_users.append(len(user2items[i]))
        # max_deg = max(max_deg, len(user2items[i]))
        # min_deg = min(min_deg, len(user2items[i]))
        deg_items.append(len(item2users[i]))
        max_deg = max(max_deg, len(item2users[i]))
        min_deg = min(min_deg, len(item2users[i]))
    data_user.append(deg_users)
    data_item.append(deg_items)

plt.boxplot(data_item,labels=datasets)
plt.title('item deg')
plt.show()