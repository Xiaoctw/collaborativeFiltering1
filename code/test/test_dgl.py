import os
from pathlib import Path
import scipy.sparse  as sp
neighbor_num = 8
dist_mea = 'occurrence'
path=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/data'+'/lastfm'
user_path = path + '/user_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea,neighbor_num)
item_path = path + '/item_mat_distance_measure_{}_neighbor_num_{}.npz'.format(dist_mea,neighbor_num)
# user_mat, item_mat = None, None
user_mat = sp.load_npz(user_path)
item_mat = sp.load_npz(item_path)

print(user_mat.nonzero()[0])

print(user_mat.data)