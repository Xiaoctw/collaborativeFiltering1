from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import scipy.sparse as sp
import torch
import world
import utils
import random
import model
import os

random.seed(2021)
import numpy as np

np.random.seed(2021)
world_config = world.world_config

config = world.config

world_config['model_name'] = 'cf_mo'
world_config['comment'] = world_config['model_name']

import register
from register import dataset

recModel = register.MODELS[world_config['model_name']](config, dataset)

weight_file = './checkpoints' + '/model_name-{}-dataset-{}-comment-{}-n_layers-2-latent_dim-64-delete_0.pth.tar'.format(
    world_config['model_name'], world_config['dataset'], world_config['comment'])

print(weight_file)
recModel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
# users_emb, items_emb = recModel.computer()
users_emb = recModel.embedding_user.weight.data
items_emb = recModel.embedding_item.weight.data
users_norm = users_emb.pow(2).sum(1).unsqueeze(1)
items_norm = items_emb.pow(2).sum(1).unsqueeze(1)
users_emb = users_emb / users_norm
items_emb = items_emb / items_norm

users_adj = dataset.getUserGraph(dense=False)
user_indices = users_adj.indices()
print(user_indices)
items_adj = dataset.getItemGraph(dense=False)
item_indices = items_adj.indices()


def batch_sum(embs, indices, batch_size=4096):
    # num = embs.shape[0]
    # dim = embs.shape[1]
    total_sum = 0
    print(indices.shape)
    for i in range(0, indices.shape[1] // batch_size):
        beg_idx = i * batch_size
        print('{}:{}'.format(beg_idx, indices.shape[1]))
        end_idx = min((i + 1) * batch_size, indices.shape[1])
        user_idx1 = indices[0][beg_idx:end_idx]
        user_idx2 = indices[1][beg_idx:end_idx]
        tem_sum = (embs[user_idx1] - embs[user_idx2]).pow(2).sum()
        total_sum += tem_sum.tolist()
    return total_sum


user_sum = batch_sum(users_emb, user_indices)
items_sum = batch_sum(items_emb, item_indices)

print('model:{},users sum:{}'.format(world_config['model_name'], user_sum))
print('model:{} items sum:{}'.format(world_config['model_name'], items_sum))
