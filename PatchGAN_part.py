import torch.nn as nn
import torch

class P_discriminator(nn.Module):

    def __init__(self, width=90, height=120, in_channels=3, out_channels=16):
        super().__init__()

        self.width = width
        self.height = height

        sequence = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
                    ]
        sequence += [
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(out_channels * 4, 1, kernel_size=4, stride=2, padding=1),]
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(
            *sequence
        )
    def forward(self, x):
        x = x.view(x.size(0), 3, self.width, self.height)
        return self.model(x)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        #只有再forward的时候会更新
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)