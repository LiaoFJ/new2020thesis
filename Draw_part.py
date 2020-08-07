import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np


class DRAWModel(nn.Module):
    def __init__(self, T, A, B, in_channels, readn, writen, params):
        super().__init__()

        self.T = T                          #25
        self.A = A                         #32
        self.B = B                          #32
        self.z_size = params['z_size']                  #200
        self.read_N = readn                  #6
        self.write_N = writen                #6
        self.enc_size = params['enc_size']              #400
        self.dec_size = params['dec_size']              #400
        self.channel = in_channels                      #3

        self.device = params['device']
        # Stores the generated image for each time step.
        self.cs = [0] * self.T
        self.c_result = [0] * self.T
        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.T
        self.sigmas = [0] * self.T
        self.mus = [0] * self.T

        self.encoder = nn.LSTMCell(2*self.read_N*self.read_N*self.channel + self.dec_size, self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
        self.fc_mu = nn.Linear(self.enc_size, self.z_size)
        self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(self.dec_size, self.write_N*self.write_N*self.channel)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)




    def read(self, x, x_hat, h_dec_prev):
        # Using attention
        (Fx, Fy), gamma = self.attn_window(h_dec_prev, self.read_N)

        def filter_img(img, Fx, Fy, gamma):
            Fxt = Fx.transpose(3, 2)

            img = img.view(-1, self.channel, self.B, self.A)
                # Equation 27.
            glimpse = torch.matmul(Fy, torch.matmul(img, Fxt))
            glimpse = glimpse.view(-1, self.read_N * self.read_N * self.channel)

            return glimpse * gamma.view(-1, 1).expand_as(glimpse)

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)

        return torch.cat((x, x_hat), dim=1)

    def write(self, h_dec):
        # Using attention
        # Equation 28.
        w = self.fc_write(h_dec)
        w = w.view(self.batch_size, self.channel, self.write_N, self.write_N)


        (Fx, Fy), gamma = self.attn_window(h_dec, self.write_N)
        Fyt = Fy.transpose(3, 2)

        # Equation 29.
        wr = torch.matmul(Fyt, torch.matmul(w, Fx))
        wr = wr.view(self.batch_size, self.B * self.A * self.channel)


        return wr / gamma.view(-1, 1).expand_as(wr)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)
        # e = torch.randn(self.batch_size, self.z_size)
        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def attn_window(self, h_dec, N):
        # Equation 21.
        params = self.fc_attention(h_dec)
        gx_, gy_, log_sigma_2, log_delta_, log_gamma = params.split(1, 1)

        # Equation 22.
        gx = (self.A + 1) / 2 * (gx_ + 1)
        # Equation 23
        gy = (self.B + 1) / 2 * (gy_ + 1)
        # Equation 24.
        delta = (max(self.A, self.B) - 1) / (N - 1) * torch.exp(log_delta_)
        sigma_2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma_2, delta, N), gamma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(start=0.0, end=N, requires_grad=True, device=self.device).view(1, -1)
        # grid_i = torch.arange(start=0.0, end=N, requires_grad=True).view(1, -1)
        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, requires_grad=True, device=self.device).view(1, 1, -1)
        b = torch.arange(0.0, self.B, requires_grad=True, device=self.device).view(1, 1, -1)
        # a = torch.arange(0.0, self.A, requires_grad=True).view(1, 1, -1)
        # b = torch.arange(0.0, self.B, requires_grad=True).view(1, 1, -1)
        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
        Fx = Fx.repeat(1, self.channel, 1, 1)

        Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
        Fy = Fy.repeat(1, self.channel, 1, 1)

        return Fx, Fy

    def forward(self, x):
        self.batch_size = x.size(0)

        x = x.view(self.batch_size, self.channel, self.B, self.A)

        x = x.view(self.batch_size, -1)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)
        # h_enc_prev = torch.zeros(self.batch_size, self.enc_size, requires_grad=True)
        # h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True)
        #
        # enc_state = torch.zeros(self.batch_size, self.enc_size, requires_grad=True)
        # dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True)

        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B * self.A * self.channel,requires_grad=True, device=self.device) if t == 0 else self.cs[t - 1]
            # c_prev = torch.zeros(self.batch_size, self.B * self.A * self.channel,requires_grad=True) if t == 0 else self.cs[t - 1]
             # Equation 3.

            x_hat = x - torch.sigmoid(c_prev)
            # Equation 4.

            r_t = self.read(x, x_hat, h_dec_prev)
            # Equation 5.

            h_enc, enc_state = self.encoder(torch.cat((r_t, h_dec_prev), dim=1), (h_enc_prev, enc_state))
            # Equation 6.

            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc)
            # Equation 7.

            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))

             # Equation 8.
            self.cs[t] = c_prev + self.write(h_dec)

            h_enc_prev = h_enc
            h_dec_prev = h_dec

        return self.cs[-1].view(self.batch_size, self.channel, self.B, self.A)


    def loss(self, x):
        self.forward(x)
        #
        # criterion = nn.MSELoss()
        # x_recon = torch.sigmoid(self.c_result[-1])
        #
        # Lx = (criterion(x_recon, x) **2) * self.A * self.B * self.channel
        # Latent loss.
        Lz = 0

        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = 0.5*torch.sum(mu_2 + sigma_2 - 2*logsigma, 1) - 0.5*self.T
            Lzsum = Lz + kl_loss
        # print(Lz)
        Lzavg = torch.mean(Lzsum)
        Lx = 0
        net_loss = Lx + Lzavg

        return net_loss

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size, device=self.device)
        # h_dec_prev = torch.zeros(num_output, self.dec_size)
        # dec_state = torch.zeros(num_output, self.dec_size)
        for t in range(self.T):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel, device=self.device) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            # c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel) if t == 0 else self.cs[t-1]
            # z = torch.randn(self.batch_size, self.z_size)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)

        return self.cs[-1].view(self.batch_size, self.channel, self.B, self.A)