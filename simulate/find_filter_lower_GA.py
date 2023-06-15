from sko.GA import GA
import os
import sys
sys.path.append('E:\pythonProject\python3\myutils_v2')
sys.path.append('E:\pythonProject\python3 v2\SpecNeRF-v2')
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from glob import glob
from filesort_int import sort_file_int
from tqdm import tqdm


def get_filters(path):
    fs = glob(f'{path}/*.mat')[1:]
    filters = list(map(lambda x: np.diagonal(sio.loadmat(x)['filter']), fs))
    filters = np.row_stack(filters)

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


