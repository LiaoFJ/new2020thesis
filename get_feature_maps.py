import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from dataloader import get_data, get_test_img_single
import os
import argparse
from Final_model_re import Final_model_re
from Final_model import  Final_model
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# Function to generate new images and save the time-steps as an animation.

# Dictionary storing network parameters.
params = {
    'batch_size': 16,  # Batch size.
    'save_epoch': 10,  # After how many epochs to save checkpoints and generate test output.
    'epoch_num': 50,  # Number of epochs to train for.
    'learning_rate': 3e-4,  # Learning rate.

}  # Number of channels for image.(3 for RGB, etc.)

#loader
train_loader = get_data(params)


#parser
parser = argparse.ArgumentParser()
parser.add_argument('-use_gpu', default='False', help='if use gpu to accelerate')
parser.add_argument('-load_if', default='False')
parser.add_argument('-load_path', default='./checkpoint/model_epoch_300.pkl', help='Checkpoint to load path from')
args = parser.parse_args()

# Initialize the model.

if args.use_gpu == 'True':
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    model = Final_model(params).to(device)
    model = nn.DataParallel(model.cuda())
else:
    device = torch.device("cpu")
params['device'] = device

#go on training
print(device, " will be used.\n")

if args.load_if == 'True':
    # Load the checkpoint file.
    print('load_path is:', args.load_path)
    state_dict = torch.load(args.load_path)
    # Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']
    model = Final_model_re(params).to(device)
    model.load_state_dict(state_dict['model'])
    step = state_dict['step']
    print('load finished and then train')
else:
    model = Final_model(params).to(device)
    step = 0

print(model)


with torch.no_grad():
    img = get_test_img_single()
    img = img.view(1, -1).to(device)
    x = model(img)
    x = x.view(x.size(0), 3, 128, 128)
    gene = x.cpu()
    save_image(gene, './result/test_img.jpg')

