import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(i_channels, o_channels, kernel_size=3),
            nn.BatchNorm2d(o_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleconv(x)

class down_conv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.downconv = nn.Sequential(
        nn.MaxPool2d(2),
        SingleConv(i_channels, o_channels),
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
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = SingleConv(i_channels, o_channels)

    def forward(self, x1, x2):
        # print('1', x1.size())
        x1 = self.up(x1)
        # print('2', x1.size())
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]


        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #对x1做padding处理
        # print(x2.size(), x1.size())
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, i_channels, o_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(i_channels, o_channels, kernel_size=3, stride=1)

    def forward(self, x):
        print(x.size())
        x = self.conv(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        return x

