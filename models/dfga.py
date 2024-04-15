import torch
import torch.nn as nn
import numpy as np
from .module import *

class DFGANet(nn.Module):

    def __init__(self, base_channels=8, channels_i=[192,72,64,10,8,4,1]):
        super(DFGANet, self).__init__()
        self.upsample1 = Deconv2d(base_channels*channels_i[0], base_channels*channels_i[2], kernel_size=4, stride=2, padding=1)
        self.upsample2 = Deconv2d(base_channels*channels_i[2], base_channels*channels_i[4], kernel_size=4, stride=2, padding=1)

        self.conv0_0 = nn.Sequential(Conv2d(base_channels*channels_i[2], base_channels*channels_i[5],kernel_size=3,stride=1, padding=1),
                                     Conv2d(base_channels*channels_i[5], base_channels*channels_i[5], kernel_size=1, stride=1, padding=0))
        self.conv0_1 = nn.Sequential(Conv2d(base_channels*channels_i[4], base_channels*channels_i[6], kernel_size=3, stride=1, padding=1),
                                     Conv2d(base_channels*channels_i[6], base_channels*channels_i[6], kernel_size=1, stride=1, padding=0))

        self.output1 = nn.Conv2d(base_channels*channels_i[1], base_channels*channels_i[1], kernel_size=1, stride=1, padding=0)
        self.output2 = nn.Conv2d(base_channels*channels_i[3], base_channels*channels_i[3], kernel_size=1, stride=1, padding=0)

    def forward(self, c, f, stage_idx):
        """forward.

        :param c: [B, {32,16}, {48,32}, H, W], coarse cost volume
        :param f: [B, {16,08}, {32,08}, H, W], fine cost volume
        """
        ori = f
        c = torch.flatten(c, 1, 2)
        f = torch.flatten(f, 1, 2)
        if stage_idx == 1:
            c = self.upsample1(c)
            c2f = self.conv0_0(c)
            f2f = self.conv0_0(f)
            c2f = c2f.unsqueeze(1)
            f2f = f2f.unsqueeze(1)
            cf_cost_1 = torch.concat((c2f, f2f, ori),1)
            cf_cost = torch.flatten(cf_cost_1, 1, 2)
            final_cost = self.output1(cf_cost).unsqueeze(1).view(ori.shape[0],cf_cost_1.shape[1],ori.shape[2],ori.shape[3],ori.shape[4])
        else:
            c = self.upsample2(c)
            c2f = self.conv0_1(c)
            f2f = self.conv0_1(f)
            c2f = c2f.unsqueeze(1)
            f2f = f2f.unsqueeze(1)
            cf_cost_2 = torch.concat((c2f, f2f, ori),1)
            cf_cost = torch.flatten(cf_cost_2, 1, 2)
            final_cost = self.output2(cf_cost).unsqueeze(1).view(ori.shape[0],cf_cost_2.shape[1],ori.shape[2],ori.shape[3],ori.shape[4])

        return final_cost