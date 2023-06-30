import numpy as np
from scipy.spatial.distance import pdist, squareform


# 示例用法
matrix = np.array([[1,2],[5,6],[9,10],[13,14]])
xx, yy = np.meshgrid(matrix, matrix)

comb = np.array(np.meshgrid(matrix, matrix)).T.reshape(-1, 2)
comb2 = comb[[1,2,3,6,7,11], :]
np.triu_indices
pass

# 计算列之间的欧几里得距离
# distances = pdist(matrix.T, metric='euclidean')

# # 将距离转换为相似度
# similarities = 1 / (1 + squareform(distances))

# print(similarities)
