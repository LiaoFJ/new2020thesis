import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(i_channels, o_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(o_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(o_channels, o_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(o_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.double_conv(x)

class down_conv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.downconv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(i_channels, o_channels),
        )
    def forward(self, x):
        return self.downconv(x)


class up_conv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(i_channels, o_channels, kernel_size=1)
        )
        self.conv = DoubleConv(i_channels, o_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(i_channels, o_channels, stride=1),
            nn.BatchNorm2d(o_channels),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(o_channels, o_channels, stride=1),
            nn.BatchNorm2d(o_channels)

        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x


