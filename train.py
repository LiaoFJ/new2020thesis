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
    x = model.module.generate(16)
    gene = x.cpu()
    save_image(gene, './result/test_img_generated_{}.jpg'.format(epoch))
    # fig = plt.figure(figsize=(16, 16))
    # plt.axis("off")
    # ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in x]
    # anim = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000, blit=True)
    # anim.save('./result/draw_epoch_{}.gif'.format(epoch), dpi=100, writer='pillow')
    # plt.close('all')

# Dictionary storing network parameters.
params = {
    'batch_size': 4,  # Batch size.
    'z_size': 64,  # Dimension of latent space.
    # 'read_N': 5,  # N x N dimension of reading glimpse.
    # 'write_N': 5,  # N x N dimension of writing glimpse.
    'dec_size': 100,  # Hidden dimension for decoder.
    'enc_size': 100,  # Hidden dimension for encoder.
    'epoch_num': 200,  # Number of epochs to train for.
    'learning_rate': 3e-4,  # Learning rate.
    'clip': 5.0,
    'save_epoch': 5,  # After how many epochs to save checkpoints and generate test output.
            }  # Number of channels for image.(3 for RGB, etc.)

#loader
train_loader = get_data(params)


#parser
parser = argparse.ArgumentParser()
parser.add_argument('-use_gpu', default='False', help='if use gpu to accelerate')
parser.add_argument('-load_path', default='./checkpoint/model_epoch_300.pkl', help='Checkpoint to load path from')
parser.add_argument('-load_if', default='False')
args = parser.parse_args()


# Plot the training images.
sample_batch = next(iter(train_loader))
plt.figure(figsize=(16, 16))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0][: 16], nrow=4, padding=1, normalize=True, pad_value=1).cpu(), (1, 2, 0)))
plt.savefig("Training_Data")

# Initialize the model.

if args.use_gpu == 'True':
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    params['device'] = device

else:
    device = torch.device("cpu")
    params['device'] = device


#go on training

if args.load_if == 'True':

    # Load the checkpoint file.
    state_dict = torch.load(args.load_path)

    # Get the 'params' dictionary from the loaded state_dict.
    params_add = state_dict['params']
    params.update(params_add)
    params['batch_size'] = 4
    model = Final_model(params).to(device)
    model = nn.DataParallel(model.cuda())
    model.load_state_dict(state_dict['model'], strict=False)

    step = state_dict['step']
    print('load finished and then train')
else:
    model = Final_model(params).to(device)
    step = 0

freeze_layers=['inc', 'down_1', 'down_2', 'down_3', 'up_1', 'up_2', 'up_3', 'OutConv']

print('************************')
print('freeze some layers')
for name, module in model._modules.items():
    if name in freeze_layers:
        for p in module.parameters():
            p.requires_grad = False

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['learning_rate'])
print('************************')


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
        # Calculate the loss.
        # loss = model.module.loss(data)
        loss = model.module.loss(data)
        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val
        # Calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        # Update parameters.
        optimizer.step()

        # Check progress of training.
        if i != 0 and i % 100 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f'
                  % (epoch + 1, params['epoch_num'], i, len(train_loader), avg_loss / 100))

            avg_loss = 0

        losses.append(loss_val)
        iters += 1

    avg_loss = 0
    epoch_time = time.time() - epoch_start_time
    print("Time Taken for Epoch %d: %.2fs" % (epoch + 1 + step, epoch_time))
    # Save checkpoint and generate test output.
    if (epoch + 1) % params['save_epoch'] == 0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'params': params,
            'step': epoch,
            'params_loss': losses
        }, 'checkpoint/model_epoch_{}_se.pkl'.format(epoch + 1 + step))

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
    'params': params,
    'step': params['epoch_num'],
    'params_loss': losses
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