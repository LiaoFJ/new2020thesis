import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3 * 90 * 120, 512),#np.prod(img_shape)连乘操作，长*宽*深度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),#改进1、判别器最后一层去掉sigmoid。sigmoid函数容易出现梯度消失的情况。
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

