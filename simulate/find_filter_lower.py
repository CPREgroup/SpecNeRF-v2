import os
import sys
sys.path.append('E:\pythonProject\python3\myutils_v2')
sys.path.append('E:\pythonProject\python3 v2\SpecNeRF-v2')
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from glob import glob
from filesort_int import sort_file_int
from tqdm import tqdm
from myutils import myutils


class FindFilter(nn.Module):

    def __init__(self, path) -> None:
        super(FindFilter, self).__init__()

        # self.fs = sort_file_int(f'{path}/*.mat', 'mat')[1:]
        self.fs = glob(f'{path}/*.mat')[1:]
        filters = list(map(lambda x: np.diagonal(sio.loadmat(x)['filter']), self.fs))
        self.filters = torch.from_numpy(np.row_stack(filters)).cuda()
        self.indicator = nn.Parameter(torch.rand([len(filters), 1]))

    def forward(self):
        indicator = torch.clamp(self.indicator, 1e-3, 1)
        # sumall = torch.sum(self.filters * indicator, dim=0)
        # sumall = torch.mean(self.filters * indicator, dim=0)
        # sumall = torch.prod(self.filters * indicator, dim=0)
        return self.filters, indicator


from collections import Counter

def findINgroup(v1, g):
    minus = torch.where((v1 - g) >= 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    diff = minus[:, 1:] - minus[:, :-1]
    inter_id = torch.nonzero(diff != 0)
    if inter_id.shape[0] == 0:
        return False
    else:
        return inter_id[:, 1]


def get_intersection_dist(filters):
    xs = []
    for i in range(filters.shape[0] - 1):
        x = findINgroup(filters[i, :], filters[i+1:, :])
        if x is not False:
            # y(x) should be low
            ys = filters[i, x]
            ratio = ys / filters[i, :].max()
            stay_idx = torch.nonzero((ratio < 0.5) & (ratio > 0.1) & (ys > 0.05)).flatten()
            xs.append(x[stay_idx])

    xs = torch.cat(xs)
    x, y = torch.unique(xs, return_counts=True)

    return x, y


def main():
    iter = 10000
    filterN = 19

    findfilter_net = FindFilter(r'E:\pythonProject\python3 v2\SpecNeRF-v2\myspecdata\filterset').cuda()
    optimizer = torch.optim.Adam(findfilter_net.parameters(), 0.001)


    pbar = tqdm(range(iter), miniters=500, file=sys.stdout)
    for iteration in pbar:
        filters, indicator = findfilter_net()
        filters_selected = filters * indicator

        sumall = torch.sum(filters_selected, dim=0)
        loss = torch.var(sumall)
        # summean = sumall.mean()
        # summax = sumall.max()
        # loss = (summean - 0.5).abs() * 0.1 + (summean - summax).abs()
        x, y = get_intersection_dist(filters)
        intra_var = torch.var(y.float())
        loss += intra_var


        reg = (torch.log(indicator) + torch.log(1 - indicator)).mean()
        regnum = (indicator.sum() - filterN) ** 2
        loss += reg + regnum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            f'Iteration {iteration:05d}:'
            + f' loss = {loss.detach().item()}'
            + f' intra_var = {intra_var.detach().item()}'
            + f' reg = {reg.detach().item()}'
            + f' regnum = {regnum.detach().item()}'
        )

    print(indicator)
    pickid = torch.argwhere(indicator.reshape(-1) >= 0.9).reshape(-1)
    pickfilter = [findfilter_net.fs[i][-8:] for i in pickid]
    print(pickfilter)

    plt.plot(sumall.detach().cpu().numpy() * filterN)

    randomfolder = myutils.makeStrID()
    os.mkdir(f"log_filter/{randomfolder}")
    plt.savefig(f"log_filter/{randomfolder}/res.png")
    with open(f'log_filter/{randomfolder}/res.txt', 'a+') as f:
        f.write(str(pickfilter))
    # plt.show()





if __name__ == '__main__':
    main()

