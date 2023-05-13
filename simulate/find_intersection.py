


from collections import Counter
import os
from matplotlib import pyplot as plt
import numpy as np

import scipy.io


folder_path = r'myspecdata\filter19\filters_measure' # './myspecdata/filter15_v1/filters' #  #
file_list = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

filters = []

for file_name in file_list:
    # load matrix data from .mat file
    mat_data = scipy.io.loadmat(os.path.join(folder_path, file_name))
    # extract diagonal elements of the matrix
    filters.append(np.diag(mat_data['filter']).flatten())
filters = np.row_stack(filters).astype(np.float32)


def findINgroup(v1, g):
    minus = np.where((v1 - g) >= 0, 1, 0)
    diff = minus[:, 1:] - minus[:, :-1]
    inter_id = np.argwhere(diff != 0)
    if inter_id.shape[0] == 0:
        return []
    else:
        return inter_id[:, 1].tolist()



def main():
    xs = []
    for i in range(filters.shape[0] - 1):
        x = findINgroup(filters[i, :], filters[i+1:, :])
        xs += x

    print(xs)
    counter = Counter(tuple(xs))

    # 提取元素和计数器的值作为 x 和 y 数据
    x = list(counter.keys())
    y = list(counter.values())

    # 绘制柱状图
    plt.bar(x, y)

    # 添加标签和标题
    plt.xlabel('元素')
    plt.ylabel('出现次数')
    plt.title('元素出现次数分布图')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    main()

