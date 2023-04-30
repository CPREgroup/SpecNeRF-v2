
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args
from utils import positionencoding1D

class SSFFcn(torch.nn.Module):
    def __init__(self, L, out_dim, *ags, **kags):
        super(SSFFcn, self).__init__()

        self.input_1D = torch.from_numpy(positionencoding1D(args.spec_channel, 2)).float().cuda()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2 * L + 1, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=2*L, out_features=2*L),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=out_dim),
            nn.Sigmoid()
        )

    def forward(self, *ags, **kags):

        y = self.layers(self.input_1D)

        return y
    

class SSFMatrix(torch.nn.Module):
    def __init__(self, *ags, **kags):
        super(SSFMatrix, self).__init__()

        self.ssf = nn.Parameter(torch.random((args.spec_channel, 3)))

    def forward(self, *ags, **kags):
        y = torch.sigmoid(self.ssf)

        return y
