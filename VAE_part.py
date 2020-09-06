import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self, in_channels, input_dim, second_dim, latent_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.input_dim = input_dim

        self.device = device
        self.fc1 = nn.Sequential(
            nn.Linear(self.in_channels * self.input_dim ** 2, second_dim),
            nn.LeakyReLU(0.2)
        )

        self.enc1 = nn.Linear(second_dim, self.latent_dim)

        self.enc2 = nn.Linear(second_dim, self.latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(self.latent_dim, second_dim),
            nn.LeakyReLU(0.2)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(second_dim, self.in_channels * self.input_dim ** 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        se = self.fc1(x)
        self.mu = self.enc1(se)
        self.logvar = self.enc2(se)
        se_out = self.dec(self.re_parameter(self.mu, self.logvar))
        out = self.fc_out(se_out)
        out = out.view(out.size(0), self.in_channels, self.input_dim, self.input_dim)
        return out



    def re_parameter(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        sample_z = mu + eps*std
        return sample_z

    def loss(self, x):
        recon_x = self.forward(x)
        BCE = F.binary_cross_entropy(recon_x, x.detach(), reduction='sum')
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        print('BCE: ', BCE, 'KLD: ', KLD)
        return BCE + KLD

    def generate(self, num_out):
        batch_size = num_out

        sample = torch.randn(batch_size, self.latent_dim).to(self.device)
        se_out = self.dec(sample)
        out = self.fc_out(se_out)
        return out.view(batch_size, self.in_channels, self.input_dim, self.input_dim)