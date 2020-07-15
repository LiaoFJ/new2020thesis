import torch
import torch.nn as nn
import Unet_part as U
import Draw_part as D

class Final_model(nn.Module):
    def __init__(self, params):
        '''
        :param size_a: width of image
        :param size_b: height of image
        '''
        super().__init__()
        self.inc = U.SingleConv(3, 32)
        self.down_1 = U.down_conv(32, 64)
        self.down_2 = U.down_conv(64, 128)
        self.down_3 = U.down_conv(128, 256)


        self.up_1 = U.up_conv(256, 128)
        self.up_2 = U.up_conv(128, 64)
        self.up_3 = U.up_conv(64, 32)

        self.OutConv = U.OutConv(32, 3)
        #glimpses, width, heights, channels, read_N, write_N
        self.Draw_model_1 = D.DRAWModel(64, 118, 88, 32, 15, 15, params)
        self.Draw_model_2 = D.DRAWModel(64, 57, 42, 64, 8, 8, params)
        self.Draw_model_3 = D.DRAWModel(32, 26, 19, 128, 5, 5, params)
        self.Draw_model_4 = D.DRAWModel(16, 11, 7, 256, 4, 4, params)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 120, 90)
        x_out_1 = self.inc(x)
        x_out_2 = self.down_1(x_out_1)
        x_out_3 = self.down_2(x_out_2)
        x_out_4 = self.down_3(x_out_3)

        x_model_1 = self.Draw_model_1(x_out_1)
        self.loss_1 = self.Draw_model_1.loss(x_out_1)
        x_model_2 = self.Draw_model_2(x_out_2)
        self.loss_2 = self.Draw_model_1.loss(x_out_2)
        x_model_3 = self.Draw_model_3(x_out_3)
        self.loss_3 = self.Draw_model_1.loss(x_out_3)
        x_model_4 = self.Draw_model_4(x_out_4)
        self.loss_4 = self.Draw_model_1.loss(x_out_4)

        x_model_out_3 = self.up_1(x_model_4, x_model_3)
        x_model_out_2 = self.up_2(x_model_out_3, x_model_2)
        x_model_out_1 = self.up_3(x_model_out_2, x_model_1)
        x_model_output = self.OutConv(x_model_out_1)

        return x_model_output

    def loss(self, x):
        x_recon = self.forward(x)
        latent_loss = self.loss_1 + self.loss_2 + self. loss_3 + self.loss_4
        criterion = nn.MSELoss()
        recon_loss = (criterion(x_recon, x) ** 2) * x.size()[1] * x.size()[2] * x.size()[3]


        return latent_loss + recon_loss

    def generate(self, num_output):
        x_generate_4 = self.Draw_model_4.generate(num_output)
        x_generate_3 = self.Draw_model_3.generate(num_output)
        x_generate_2 = self.Draw_model_2.generate(num_output)
        x_generate_1 = self.Draw_model_1.generate(num_output)

        x_generate_mix_3 = self.up_1(x_generate_4, x_generate_3)
        x_generate_mix_2 = self.up_1(x_generate_mix_3, x_generate_2)
        x_generate_mix_1 = self.up_1(x_generate_mix_2, x_generate_1)

        img = self.OutConv(x_generate_mix_1)
        return img