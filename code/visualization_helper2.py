import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import scipy.sparse as sp

import world
import utils
import random
import model
import os

random.seed(2021)
import numpy as np

np.random.seed(2021)
world_config = world.world_config

world_config['model_name'] = 'lgn'
world_config['comment'] = world_config['model_name']
config = world.config
import register
from register import dataset

recModel = register.MODELS[world_config['model_name']](config, dataset)

weight_file = './checkpoints' + '/model_name-{}-dataset-{}-comment-{}-n_layers-2-latent_dim-64-delete_0.pth.tar'.format(
    world_config['model_name'], world_config['dataset'], world_config['comment'])


recModel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
users_emb, items_emb = recModel.computer()
# users_emb=recModel.embedding_user.weight.data
# items_emb=recModel.embedding_item.weight.data
# users_norm = users_emb.pow(2).sum(1).sqrt().unsqueeze(1)
# users_emb = users_emb / users_norm
# items_norm = items_emb.pow(2).sum(1).sqrt().unsqueeze(1)
# items_emb = items_norm / items_norm


users = [150, 250, 350, 450]

# adj_user = [74, 468, 855, 863, 1007, 1167, 1406, 1827, 23, 67, 108, 354, 426, 555, 667, 1069,
#             151, 368, 514, 736, 1071, 1678, 1781, 1850]

# embs = np.concatenate((users_emb[users], users_emb[adj_user]), axis=0)
top_k = 30
ratings = torch.mm(users_emb[users], items_emb.t())
_, idxes = torch.topk(ratings, dim=1, k=top_k)
idxes = idxes.view(-1)
embs = torch.cat([users_emb[users], items_emb[idxes]], dim=0)
embs = embs.detach().numpy()

# transfer = PCA(n_components=2)
# embed = transfer.fit_transform(embs)

tsne = TSNE(n_components=2, verbose=1, init='random')
embed = tsne.fit_transform(embs)

plt.scatter(embed[:len(users), 0], embed[:len(users), 1], c='b', marker='v', label='users')
plt.scatter(embed[len(users):len(users) + top_k, 0], embed[len(users):len(users) + top_k, 1], c='r',
            label='items')
plt.scatter(embed[len(users) + top_k:len(users) + 2 * top_k, 0], embed[len(users) + top_k:len(users) + 2 * top_k, 1],
            c='g', label='items')
plt.scatter(embed[len(users) + 2 * top_k:len(users) + 3 * top_k, 0],
            embed[len(users) + 2 * top_k:len(users) + 3 * top_k, 1], c='y', label='items')
plt.scatter(embed[len(users) + 3 * top_k:len(users) + 4 * top_k, 0],
            embed[len(users) + 3 * top_k:len(users) + 4 * top_k, 1], c='c', label='items')
plt.title('{} {}'.format(world_config['model_name'],world_config['dataset']))
plt.legend(loc='upper right')

picture_path = '../pictures' + '/{} {}.png'.format(world_config['comment'], world_config['dataset'])
plt.savefig(picture_path)
plt.show()
