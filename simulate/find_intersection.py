


from collections import Counter
import os
from tkinter import getint
from matplotlib import pyplot as plt
import numpy as np

import scipy.io


def get_filters():
    folder_path = r'myspecdata\filters19_optimized\filters_interp25' # r'myspecdata\filter19\filters_measure\sparse_sampled' # r'myspecdata\filters19_optimized\filters_interp25' # './myspecdata/filter15_v1/filters' #  #
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    filters = []

    for file_name in file_list:
        # load matrix data from .mat file
        mat_data = scipy.io.loadmat(os.path.join(folder_path, file_name))
        # extract diagonal elements of the matrix
        filters.append(np.diag(mat_data['filter']).flatten())
    filters = np.row_stack(filters).astype(np.float32)

    return filters


def findINgroup(v1, g):
    minus = np.where((v1 - g) >= 0, 1, 0)
    diff = minus[:, 1:] - minus[:, :-1]
    inter_id = np.argwhere(diff != 0)
    if inter_id.shape[0] == 0:
        return np.array([])
    else:
        return inter_id[:, 1]


def get_intersection_dist(filters):
    xs = []
    for i in range(filters.shape[0] - 1):
        x = findINgroup(filters[i, :], filters[i+1:, :])
        # y(x) should be low
        ys = filters[i, x.tolist()]
        ratio = ys / filters[i, :].max()
        stay_idx = np.argwhere((ratio < 0.5) & (ratio > 0.1) & (ys > 0.05)).flatten()
        xs += x[stay_idx].tolist()

    print(xs)
    counter = Counter(tuple(xs))

    # 提取元素和计数器的值作为 x 和 y 数据
    x = list(counter.keys())
    y = list(counter.values())

    return x, y


def main():
    x, y = get_intersection_dist(get_filters())

    # 绘制柱状图
    plt.bar(x, y)

    # 添加标签和标题
    plt.xlabel('band')
    plt.ylabel('number')
    plt.title('distribution of filters intersections')

    # 显示图形
    plt.show()


if __name__ == '__main__':
    main()

