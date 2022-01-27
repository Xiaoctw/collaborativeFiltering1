import numpy as np

import torch

mat = np.array([[1, 0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0, ],
                [1, 0, 1, 1, 0, 0, 0, 0]])

d1 = np.diag(np.power(np.sum(mat, axis=0), -0.5))
d2 = np.diag(np.power(np.sum(mat, axis=1), -0.5))
mat = np.dot(np.dot(d2, mat), d1)
mat_init = mat.copy()

# print(mat)

res = np.zeros(mat.shape)
for i in range(10):
    print('-----------{}-----------'.format(i))
    print(res)
    res += mat
    mat = np.dot(mat, mat)

print(mat_init)
mat1 = np.eye(mat.shape[0]) - mat_init
print(mat1)
mat1 = np.linalg.inv(mat1)
print(mat1)
# # mat1[mat1 == 0] = 1
# mat1 = np.power(mat1, -1)
# mat1[np.isinf(mat1)] = 0
# mat1[np.isnan(mat1)] = 0
# print('--------')
print(np.dot(mat_init, mat1))
