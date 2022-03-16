import world
import utils
from world import cprint
import torch
import numpy as np
import model
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

world_config = world.world_config
config = world.config

if not os.path.exists(world_config['FILE_PATH']):
    os.makedirs(world_config['FILE_PATH'], exist_ok=True)

utils.set_seed(world_config['seed'])
print(">>SEED:", world_config['seed'])

world_config['comment'] = 'cf_mo'
world_config['model_name'] = 'cf_mo'

from register import dataset

print('---recModel---')
recModel = model.CLAGL(config, dataset)
print('--recModel finished---')

recModel = recModel.to(world_config['device'])

weight_file = utils.getFileName()
print('load and save to {}'.format(weight_file))
users = []
if world_config['dataset'] == 'lastfm':
    users = [20, 100, 200, 400, 600, 800, 1000, 1200, 1500]
elif world_config['dataset'] == 'yelp2018':
    users = [100, 200, 800, 1000, 2000, 4000, 8000, 10000, 12000, 15000, 18000, 20000]


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world_config['topks']:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


try:
    recModel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    cprint(f"loaded model weights from {weight_file}")
except FileNotFoundError:
    print(f"{weight_file} not exists, start from beginning")

results = {'precision': np.zeros(len(world_config['topks'])),
           'recall': np.zeros(len(world_config['topks'])),
           'ndcg': np.zeros(len(world_config['topks']))}
testDict = dataset.testDict
recModel = recModel.eval()
max_K = max(world_config['topks'])

with torch.no_grad():
    users_list = []
    rating_list = []
    groundTrue_list = []
    allPos = dataset.getUserPosItems(users)
    # print("allPos",allPos)
    groundTrue = [testDict[u] for u in users]
    # users_gpu = torch.Tensor(users).long().to(world_config['device'])
    users = dataset.getDeleteUserGraph(users)
    # print(recModel.embedding_item.weight.shape)
    users_emb = torch.matmul(users, recModel.embedding_item.weight)
    # rating = recModel.getUsersRating(users_gpu)
    _, all_items = recModel.computer()
    # users_emb=torch.matmul(users,all_items)
    rating = torch.matmul(users_emb, all_items.t())
    exclude_index = []
    exclude_items = []

    for range_i, items in enumerate(allPos):
        exclude_index.extend([range_i] * len(items))
        exclude_items.extend(items)
    rating[exclude_index, exclude_items] = -(1 << 10)
    _, rating_K = torch.topk(rating, k=max_K)
    rating = rating.cpu().numpy()
    del rating
    users_list.append(users)
    rating_list.append(rating_K.cpu())
    groundTrue_list.append(groundTrue)

# with torch.no_grad():



X = zip(rating_list, groundTrue_list)
pre_results = []
for x in X:
    pre_results.append(test_one_batch(x))

for result in pre_results:
    results['recall'] += result['recall']
    results['precision'] += result['precision']
    results['ndcg'] += result['ndcg']
results['recall'] /= float(len(users))
results['precision'] /= float(len(users))
results['ndcg'] /= float(len(users))

print('recall:{},precision:{},ndcg:{}'.format(results['recall'], results['precision'],
                                              results['ndcg']))
