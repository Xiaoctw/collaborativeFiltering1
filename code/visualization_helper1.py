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

recModel = model.CLAGL(config, dataset)
weight_file = './checkpoints' + '/model_name-cf_mo-dataset-{}-comment-cf_mo-n_layers-2-latent_dim-64-delete_0.pth.tar'.format(
    world_config['dataset'])

recModel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
users_emb, items_emb = recModel.computer()
# users_emb=recModel.embedding_user.weight.data
# items_emb=recModel.embedding_item.weight.data
# users_norm = users_emb.pow(2).sum(1).sqrt().unsqueeze(1)
# users_emb = users_emb / users_norm
# items_norm = items_emb.pow(2).sum(1).sqrt().unsqueeze(1)
# items_emb = items_norm / items_norm
users_emb = users_emb.detach().numpy()
items_emb = items_emb.detach().numpy()

users = [100, 200, 300]

adj_user = [74, 468, 855, 863, 1007, 1167, 1406, 1827, 23, 67, 108, 354, 426, 555, 667, 1069,
            151, 368, 514, 736, 1071, 1678, 1781, 1850]
embs = np.concatenate((users_emb[users], users_emb[adj_user]), axis=0)

# transfer=PCA(n_components=2)
# embs=transfer.fit_transform(embs)

tsne = TSNE(n_components=2, verbose=1, init='random')
embed = tsne.fit_transform(embs)
plt.scatter(embed[:, 0], embed[:, 1], c=[0] * len(users) + [1] * 8 + [2] * 8 + [3] * 8)
plt.show()
