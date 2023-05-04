import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args
from utils import positionencoding1D


class FilterCompensateNet(torch.nn.Module):
    def __init__(self, encoderHidden=128):
        super(FilterCompensateNet, self).__init__()

        # self.pe = torch.from_numpy(positionencoding1D(args.spec_channel_compensate, L)).float().cuda()

        self.filter_encoder = nn.Sequential(
            nn.Linear(in_features=2 * args.spec_channel - 1, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=args.spec_wholechannel - args.spec_channel),
            nn.Sigmoid()
        )


    def forward(self, filters):
        fin = torch.cat([filters, filters[:, 1:] - filters[:, :-1]], 1)
        redudent_filter = self.filter_encoder(fin)

        return redudent_filter

