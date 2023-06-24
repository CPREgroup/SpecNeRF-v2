import argparse

import matplotlib.pyplot as plt
import numpy as np

from filesort_int import *
import torch
from torch import nn
import scipy.io as sio

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_filters(args):
    files = sort_file_int(f'{args.filter_dir}/*.mat', 'mat')
    filters = []
    for f in files:
        filters.append(
            np.diagonal(sio.loadmat(f)['filter'])
        )

    filters = torch.FloatTensor(filters).unsqueeze(0).repeat([args.angles, 1, 1])

    return filters.permute([2, 0, 1])



class FindNet(torch.nn.Module):
    def __init__(self, args):
        super(FindNet, self).__init__()

        self.args = args
        self.cols = args.angles
        self.S = read_filters(args).to(device)
        self.filt_num = self.S.shape[-1]
        self.M = nn.Parameter(torch.randn(self.S.shape[1:]))
        pass


    def mySigmoid(self, x):
        return 1 / (1+ torch.exp(-20*(x-(0.5 + self.args.eps))))


    def forward(self,x):
        M = self.mySigmoid(self.M)
        Sm = self.S * M # 31, 12, 20

        # temp_mul = torch.ones([31, self.filt_num]).to(device)
        # for i in range(self.cols):
        #     temp_mul = temp_mul * Sm[:, i, :]

        Sm_pre, Sm_last = Sm[:, :-1, :], Sm[:, -1, :].unsqueeze(1)
        Sm_shift = torch.cat([Sm_last, Sm_pre], dim=1)
        temp_mul = Sm * Sm_shift


        L = torch.sum(temp_mul)
        return L, M



def maind(args):
    mynet = FindNet(args).to(device)
    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

    allloss = []

    for ep in range(args.epoch):
        L, M = mynet(0)

        U, S, Vh = torch.linalg.svd(M, full_matrices=True)
        highrank_loss = 1 / (1e-4 * torch.sum(S)) * 0.02

        sparse_loss = torch.linalg.norm(M,ord=1,dim=(0,1))

        loss = L + highrank_loss + sparse_loss
        allloss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 20 == 0:
            print(f'=={ep}: ', 'lossall', loss.item(), 'L', L.item(),
                  'sparse_loss', sparse_loss.item(), 'highrank_loss', highrank_loss.item())

    M = M.cpu().detach().numpy()
    M = np.where(M>0.5, 1, 0)
    print(np.sum(M))
    sio.savemat(f'find_filter_res/res_2by2_3.mat', {'mask': M})
    plt.imshow(M)
    plt.show()
    plt.plot(allloss)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_dir', type=str, default='myspecdata/filter20_no1/filters')
    parser.add_argument('--angles', type=int, default=12)
    parser.add_argument('--epoch', type=int, default=20000)
    parser.add_argument('--eps', type=float, default=0.5, help='the larger it is, the less ones')
    args = parser.parse_args()

    maind(args)

