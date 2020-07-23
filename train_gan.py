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
from WGAN_part import Discriminator
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
# Function to generate new images and save the time-steps as an animation.
def generate_image(epoch):
    x = model.generate(16)
    fig = plt.figure(figsize=(16, 16))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
    anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    anim.save('draw_epoch_{}.gif'.format(epoch), dpi=100, writer='pillow')
    plt.close('all')

# Dictionary storing network parameters.
params = {
    'batch_size': 32,  # Batch size.
    'z_size': 50,  # Dimension of latent space.
    # 'read_N': 5,  # N x N dimension of reading glimpse.
    # 'write_N': 5,  # N x N dimension of writing glimpse.
    'dec_size': 100,  # Hidden dimension for decoder.
    'enc_size': 100,  # Hidden dimension for encoder.
    'epoch_num': 200,  # Number of epochs to train for.
    'learning_rate': 2e-3,  # Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch': 10,  # After how many epochs to save checkpoints and generate test output.
    'mix_channel': 32,
    'clip_D': 0.01
            }  # Number of channels for image.(3 for RGB, etc.)

#loader
train_loader = get_data(params)

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

model = Final_model(params)
model_D = Discriminator()

# RMSprop Optimizer
optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'])
optimizer_D = optim.RMSprop(model_D.parameters(), lr = params['learning_rate'])
# List to hold the losses for each iteration.
# Used for plotting loss curve.
losses = []
iters = 0
avg_loss = 0

print("-" * 25)
print("Starting Training Loop...\n")
print('Epochs: %d\nBatch Size: %d\nLength of Data Loader: %d' % (
params['epoch_num'], params['batch_size'], len(train_loader)))
print("-" * 25)

start_time = time.time()

for epoch in range(params['epoch_num']):
    epoch_start_time = time.time()

    for i, (data, _) in enumerate(train_loader, 0):
        # Get batch size.
        bs = data.size(0)
        # Flatten the image.
        data = data.view(bs, -1).to(device)
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        # Calculate the loss.
        # loss = model.module.loss(data)


        loss = model.loss(data)
        loss_dis = -torch.mean(model_D(data)) + torch.mean(model_D(model.generate(params['batch_size'])))
        loss_recon = 0.01 * model.recon_los(data) - loss_dis

        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val
        # Calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters.
        optimizer.step()

        loss_recon.backward()
        loss_dis.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_D'])
        optimizer_D.step()

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch + 1, params['epoch_num'], i, len(train_loader), avg_loss / 100))

            avg_loss = 0

        losses.append(loss_val)
        iters += 1

    avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1, epoch_time))
    # Save checkpoint and generate test output.
    if (epoch + 1) % params['save_epoch'] == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
            'model_D': model_D.state_dict()
        }, 'checkpoint/model_epoch_{}.pkl'.format(epoch + 1))

        with torch.no_grad():
            generate_image(epoch + 1)

training_time = time.time() - start_time
print("-" * 50)
print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
print("-" * 50)
# Save the final trained network paramaters.
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'params': params
}, 'checkpoint/model_final.pkl'.format(epoch))

# Generate test output.
with torch.no_grad():
    generate_image(params['epoch_num'])

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")