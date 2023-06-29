import numpy as np
from scipy.spatial.distance import pdist, squareform
import numpy as np
from multiprocessing import Pool

def process_row(row):
    # 对行进行处理的函数
    # 这里只是简单示例，将行中的元素都乘以 2
    return row * 2

if __name__ == '__main__':
    # 创建一个示例二维数组
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 创建一个进程池
    pool = Pool()

    # 使用进程池的 map 方法来并行处理每一行
    result = pool.map(process_row, array)

    # 关闭进程池
    pool.close()
    pool.join()

    # 输出处理后的结果
    print(result)

# 示例用法
# matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# idx = np.array([[0,0],[1,1],[2,2],[3,3]])

# chose = matrix[idx[:, 0], idx[:, 1]]
# pass

# comb = np.array(np.meshgrid(matrix, matrix)).T.reshape(-1, 2)
# comb2 = comb[[1,2,3,6,7,11], :]
# np.triu_indices
# pass

# 计算列之间的欧几里得距离
# distances = pdist(matrix.T, metric='euclidean')

# # 将距离转换为相似度
# similarities = 1 / (1 + squareform(distances))

# print(similarities)
