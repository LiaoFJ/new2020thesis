import torch
import torch.nn as nn
import Unet_part as U
import Draw_part as D
import VAE_part as V
import torch.nn.functional as F
class Final_model(nn.Module):
    def __init__(self, params):
        '''
        :param size_a: width of image
        :param size_b: height of image
        '''
        super().__init__()
        self.inc = U.DoubleConv(3, 16)
        self.down_1 = U.down_conv(16, 32)
        self.down_2 = U.down_conv(32, 64)
        self.down_3 = U.down_conv(64, 128)
        self.device = params['device']

        self.up_1 = U.up_conv(128, 64)
        self.up_2 = U.up_conv(64, 32)
        self.up_3 = U.up_conv(32, 16)

        self.OutConv = U.OutConv(16, 3)
        #glimpses, width, heights, channels, read_N, write_N
        # self.Draw_model_1 = D.DRAWModel(32, 96, 96, 16, 10, 10, params)
        # self.Draw_model_2 = D.DRAWModel(16, 44, 44, 32, 6, 6, params)
        # self.Draw_model_3 = D.DRAWModel(16, 18, 18, 64, 4, 4, params)
        # self.Draw_model_4 = D.DRAWModel(8, 5, 5, 128, 2, 2, params)


        self.Draw_model_1 = V.VAE(16, 64, 3500, 300, self.device)
        self.Draw_model_2 = V.VAE(32, 32, 3500, 200, self.device)
        self.Draw_model_3 = V.VAE(64, 16, 2000, 200, self.device)
        self.Draw_model_4 = V.VAE(128, 8, 1000, 100, self.device)



    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 64, 64)
        x_out_1 = self.inc(x)
        # print('x_1 :', x_out_1.size())
        x_out_2 = self.down_1(x_out_1)
        x_out_3 = self.down_2(x_out_2)
        x_out_4 = self.down_3(x_out_3)

        x_model_1 = self.Draw_model_1(x_out_1)
        # print('x_model_1 :', x_model_1.size())
        self.loss_1 = self.Draw_model_1.loss(x_out_1)
        x_model_2 = self.Draw_model_2(x_out_2)
        self.loss_2 = self.Draw_model_2.loss(x_out_2)
        x_model_3 = self.Draw_model_3(x_out_3)
        self.loss_3 = self.Draw_model_3.loss(x_out_3)
        x_model_4 = self.Draw_model_4(x_out_4)
        self.loss_4 = self.Draw_model_4.loss(x_out_4)


        x_model_out_3 = self.up_1(x_model_4, x_model_3)
        x_model_out_2 = self.up_2(x_model_out_3, x_model_2)
        # print('x_', x_model_out_2.size())
        x_model_out_1 = self.up_3(x_model_out_2, x_model_1)
        # print('x_out', x_model_out_1.size())
        self.for_dis = x_model_out_1
        x_model_output = self.OutConv(x_model_out_1)

        return torch.sigmoid(x_model_output)

    def loss(self, x):
        x_recon = self.forward(x)
        latent_loss = self.loss_1 + self.loss_2 + self. loss_3 + self.loss_4 * 8
        criterion = nn.MSELoss()
        recon_loss = (criterion(x_recon, x) * 2) * x_recon.size(-1) *100


        return latent_loss + recon_loss

    def recon_los(self, x):
        x_recon = self.forward(x)
        criterion = nn.MSELoss()
        return (criterion(x_recon, x) * 2) * x.size(-1) * 100

    def generate(self, num_output):
        x_generate_4 = self.Draw_model_4.generate(num_output)
        x_generate_3 = self.Draw_model_3.generate(num_output)
        x_generate_2 = self.Draw_model_2.generate(num_output)
        x_generate_1 = self.Draw_model_1.generate(num_output)

        x_generate_mix_3 = self.up_1(x_generate_4, x_generate_3)
        x_generate_mix_2 = self.up_2(x_generate_mix_3, x_generate_2)
        x_generate_mix_1 = self.up_3(x_generate_mix_2, x_generate_1)

        img = self.OutConv(x_generate_mix_1)
        # batch_size, channel, height, width
        img = img.view(img.size(0), 3, 64, 64)
        return torch.sigmoid(img)