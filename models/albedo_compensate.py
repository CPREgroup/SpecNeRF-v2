import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args
from utils import positionencoding1D


class AlbedoCompensatePhi(torch.nn.Module):
    def __init__(self, encoderHidden=64, L=2):
        super(AlbedoCompensatePhi, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor([0.1]).cuda())
        self.pe = torch.from_numpy(positionencoding1D(args.spec_channel_compensate, L)).float().cuda()

        self.filter_encoder = nn.Sequential(
            nn.Linear(in_features=args.spec_channel, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=args.spec_channel_compensate),
            nn.Sigmoid()
        )

        self.ssf_encoder = nn.Sequential(
            nn.Linear(in_features=2 * L + 1, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=2*L),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=2*L, out_features=3),
            nn.Sigmoid()
        )



    def forward(self, filters):
        redudent_filter = self.filter_encoder(filters)
        redudent_ssf = self.ssf_encoder(self.pe)

        return redudent_filter, redudent_ssf

