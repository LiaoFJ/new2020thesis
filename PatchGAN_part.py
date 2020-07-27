import torch.nn as nn


class P_discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.BN = nn.BatchNorm2d(out_channels)
        sequence = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
                    ]
        sequence += [
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            self.BN,
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=4, stride=2, padding=1),
            self.BN,
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(out_channels * 4, 1, kernel_size=4, stride=2, padding=1),]
        self.model = nn.Sequential(
            *sequence
        )
    def forward(self, x):
        return self.model(x)