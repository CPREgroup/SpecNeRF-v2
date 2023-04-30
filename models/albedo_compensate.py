import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args
from utils import positionencoding1D
class AlbedoCompensate(torch.nn.Module):
    def __init__(self, encoderHidden=16, compensaterHidden=64):
        super(AlbedoCompensate, self).__init__()

        self.rgb_encoder = nn.Sequential(
            nn.Linear(in_features=3, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=args.spec_channel),
            nn.LeakyReLU(inplace=True)
        )

        self.resblock_compensater = nn.Sequential(
            nn.Linear(in_features=args.spec_channel, out_features=compensaterHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=compensaterHidden, out_features=compensaterHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=compensaterHidden, out_features=compensaterHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=compensaterHidden, out_features=3),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, rgb, spec, filters):
        rgb_encoded = self.rgb_encoder(rgb.detach())
        residual = self.resblock_compensater((rgb_encoded + spec.detach() + filters) / 3)

        return residual



class AlbedoCompensatePhi(torch.nn.Module):
    def __init__(self, encoderHidden=64, dimHidden=6):
        super(AlbedoCompensatePhi, self).__init__()

        self.pe = torch.from_numpy(positionencoding1D(args.spec_channel, 16)).float().mean(dim=-1).cuda()

        self.spec_encoder = nn.Sequential(
            nn.Linear(in_features=args.spec_channel, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=dimHidden),
            nn.LeakyReLU(inplace=True)
        )

        self.ssf_encoder = nn.Sequential(
            nn.Linear(in_features=args.spec_channel, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=encoderHidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoderHidden, out_features=dimHidden),
            nn.Sigmoid()
        )


    def forward(self, spec_filtered, ssf):
        redudent_spec = self.spec_encoder((spec_filtered.detach() + self.pe) / 2)
        redudent_ssf = self.ssf_encoder((ssf.T + self.pe) / 2)

        resd_spec = redudent_spec @ redudent_ssf.T

        return resd_spec

