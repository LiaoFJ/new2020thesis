import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils
import torch.nn as nn
from torchvision import datasets, transforms
from Final_model import Final_model
from dataloader import get_data
import os
import argparse
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

# Function to generate new images and save the time-steps as an animation.
def generate_image(epoch):
    x = model.generate(16)
    gene = x.cpu()
    save_image(gene, './result/test_img_generated_{}.jpg'.format(epoch))

# Dictionary storing network parameters.
params = {
    'batch_size': 4,  # Batch size.
    'z_size': 64,  # Dimension of latent space.
    # 'read_N': 5,  # N x N dimension of reading glimpse.
    # 'write_N': 5,  # N x N dimension of writing glimpse.
    'dec_size': 100,  # Hidden dimension for decoder.
    'enc_size': 100,  # Hidden dimension for encoder.
    'epoch_num': 50,  # Number of epochs to train for.
    'learning_rate': 3e-4,  # Learning rate.
    'clip': 5.0,
    'save_epoch': 5,  # After how many epochs to save checkpoints and generate test output.
            }  # Number of channels for image.(3 for RGB, etc.)

#loader
train_loader = get_data(params)
#parser
parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='./checkpoint/model_epoch_final.pkl', help='Checkpoint to load path from')
args = parser.parse_args()

# Initialize the model.

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
params['device'] = device


# Load the checkpoint file.
state_dict = torch.load(args.load_path)

# Get the 'params' dictionary from the loaded state_dict.
params_add = state_dict['params']
params.update(params_add)
params['batch_size'] = 4
model = Final_model(params).to(device)
model.load_state_dict(state_dict['model'], strict=False)
step = state_dict['step']
print('load finished and then get image')

# Generate test output.
with torch.no_grad():
    generate_image(params['epoch_num'])

