import torch
import torch.nn as nn
import Unet_part as U
import Draw_part as D
import VAE_part as V
import torch.nn.functional as F
class Final_model_re(nn.Module):
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


        self.up_1 = U.up_conv(128, 64)
        self.up_2 = U.up_conv(64, 32)
        self.up_3 = U.up_conv(32, 16)

        self.OutConv = U.OutConv(16, 3)


    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 64, 64)
        x_out_1 = self.inc(x)
        x_out_2 = self.down_1(x_out_1)
        x_out_3 = self.down_2(x_out_2)
        x_out_4 = self.down_3(x_out_3)

        x_model_out_3 = self.up_1(x_out_4, x_out_3)
        x_model_out_2 = self.up_2(x_model_out_3, x_out_2)
        x_model_out_1 = self.up_3(x_model_out_2, x_out_1)
        self.for_dis = x_model_out_1
        x_model_output = self.OutConv(x_model_out_1)
        return torch.sigmoid(x_model_output)

    def loss(self, x):
        x_recon = self.forward(x)
        criterion = nn.MSELoss()
        recon_loss = (criterion(x_recon, x) * 2) * x_recon.size(-1)

        return recon_loss



