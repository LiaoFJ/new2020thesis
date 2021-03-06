import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils
import argparse
import torch.nn as nn
from torchvision import datasets, transforms
from Final_model import Final_model
from PatchGAN_part import GANLoss
from dataloader import get_data, get_train_data_set
import os
from WGAN_part import Discriminator
from PatchGAN_part import P_discriminator
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# Function to generate new images and save the time-steps as an animation.

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', default='./data', help='where the data comes from')
parser.add_argument('-load_path', default='./checkpoint/model_epoch_300.pkl', help='Checkpoint to load path from')
parser.add_argument('-load_if', default='False')
args = parser.parse_args()

# Dictionary storing network parameters.
params = {
    'batch_size': 8,  # Batch size.
    'z_size': 50,  # Dimension of latent space.
    # 'read_N': 5,  # N x N dimension of reading glimpse.
    # 'write_N': 5,  # N x N dimension of writing glimpse.
    'dec_size': 100,  # Hidden dimension for decoder.
    'enc_size': 100,  # Hidden dimension for encoder.
    'epoch_num': 200,  # Number of epochs to train for.
    'learning_rate': 2e-4,  # Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch': 5,  # After how many epochs to save checkpoints and generate test output.
    'mix_channel': 32,

            }  # Number of channels for image.(3 for RGB, etc.)

#loader
train_loader = get_data(params)

# train_loader_train = get_train_data_set(args.data_path, params)




# Plot the training images.
sample_batch = next(iter(train_loader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0][: 16], nrow=4, padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig("Training_Data")

# Initialize the model.
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device("cpu")
print(device, " will be used.\n")
params['device'] = device

model = Final_model(params).to(device)
if device == "cuda:0":
    model = nn.DataParallel(model.cuda())

print('load_path is:', args.load_path)
print('if load: ', args.load_if)

if args.load_if == 'True':
    print("-" * 25)
    print('start loading')
    print("-" * 25)
    # Load the checkpoint file.
    state_dict = torch.load(args.load_path)


    # Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']
    model = Final_model(params).to(device)
    model.load_state_dict(state_dict['model'])
    model_D = P_discriminator()
    model_D.load_state_dict(state_dict['model_D'])
    step = state_dict['step']
    print('load finished and then train')
    print("-" * 25)
else:
    print("-" * 25)
    print('start init')
    model = Final_model(params).to(device)
    model_D = P_discriminator().to(device)
    step = 0

# RMSprop Optimizer
optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'])
optimizer_D = optim.RMSprop(model_D.parameters(), lr = params['learning_rate'])
# List to hold the losses for each iteration.
# Used for plotting loss curve.

losses = []
avg_loss = 0
avg_loss_D = 0
criterionGAN = GANLoss().to(device)

print("-" * 25)
print("Starting Training Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (
params['epoch_num'], params['batch_size'], len(train_loader)))
print("-" * 25)

start_time = time.time()

for epoch in range(params['epoch_num']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(train_loader, 0):
        print(data.size())
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)



        #loss of generator
        # loss = model.module.loss(data)
        loss = model.loss(data)
        #loss of discriminator
        loss_d_fake = criterionGAN(model_D(model.generate(params['batch_size'])), False)
        loss_d_real = criterionGAN(model_D(data), True)
        loss_dis = (loss_d_fake + loss_d_real) * 0.5

        print('params1: ', model.recon_los(data), 'loss_d_fake: ', loss_d_fake, 'loss_d_real: ', loss_d_real)
        loss_recon = 0.000001 * model.recon_los(data) + loss_dis
        #10 times scale: loss_recon to loss_d_fake
        loss_val_G = loss.cpu().data.numpy()
        loss_val_D = loss_dis.cpu().data.numpy()


        avg_loss += loss_val_G
        avg_loss_D += loss_val_D

        # Calculate the gradients.

        #generator update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        optimizer.step()
        loss_recon.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        optimizer.step()

        if loss_d_real > 0.1 :
            #discriminator update
            optimizer_D.zero_grad()
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(model_D.parameters(), params['clip'])
            optimizer_D.step()

        # Check progress of training.

        print('[%d/%d][%d/%d]\tLoss: %.4f'
            % (epoch + 1, params['epoch_num'], i, len(train_loader), avg_loss / 5))
        print('loss D:', avg_loss_D / 5)

        avg_loss = 0
        avg_loss_D = 0
        losses.append(loss_val_G)


    avg_loss = 0
    avg_loss_D = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + step + 1, epoch_time))
    # Save checkpoint and generate test output.
    if (epoch + 1) % params['save_epoch'] == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
            'model_D': model_D.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'step': epoch + 1 + step
        }, 'checkpoint/model_epoch_{}_gan.pkl'.format(epoch + 1 + step))

        # with torch.no_grad():
        #     generate_image(epoch + 1)

training_time = time.time() - start_time
print("-" * 50)
print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
print("-" * 50)
# Save the final trained network paramaters.
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'params': params,
    'model_D': model_D.state_dict(),
    'optimizer_D': optimizer_D.state_dict()
}, 'checkpoint/model_final.pkl'.format(epoch))


plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")