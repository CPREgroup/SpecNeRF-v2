import torch
import torch.nn as nn
import torch.nn.functional as F
from opt import args

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


