import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
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
config = world.config
from register import dataset

rec_model = model.CF_MO(config, dataset)

weight_file = os.path.dirname(__file__) + '/checkpoints' + '/cf_mo-lastfm-cf_mo-2-64.pth.tar'

rec_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
# embed_item = model['embedding_item.weight'].numpy()
# embed_user = model['embedding_user.weight'].numpy()
embed_user, embed_item = rec_model.computer()
embed_user = embed_user.detach().numpy()
embed_item = embed_item.detach().numpy()
embed = np.concatenate([embed_user, embed_item], axis=0)
label = [0] * embed_user.shape[0] + [1] * embed_item.shape[0]

tsne = TSNE(n_components=2, verbose=1)
embed = tsne.fit_transform(embed)
# embed = PCA(n_components=2).fit_transform(embed)
plt.scatter(embed[:, 0], embed[:, 1])
plt.legend()
plt.show()