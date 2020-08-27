import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import torchvision.utils as vutils
import torch.nn as nn
from torchvision import datasets, transforms
from dataloader import get_data, get_test_img_single
import os
import argparse
from Final_model_re import Final_model_re
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
    model = Final_model_re(params).to(device)
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
    model = Final_model_re(params)
    model.load_state_dict(state_dict['model'])
    step = state_dict['step']
    print('load finished and then train')
else:
    model = Final_model_re(params).to(device)
    step = 0

optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])



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
        loss = model.loss(data)
        loss_val = loss.cpu().data.numpy()
        avg_loss += loss_val
        # Calculate the gradients.
        loss.backward()
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
        }, 'checkpoint/conv_weight_{}.pkl'.format(epoch + 1 + step))



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
}, 'checkpoint/conv_final.pkl'.format(epoch))

# Generate test output.



with torch.no_grad():
    img = get_test_img_single()
    img = img.view(1, -1).to(device)
    x = model(img)
    print(x.size())
    gene = x.cpu().data.numpy()
    ims = plt.imshow(np.transpose(gene, (1, 2, 0)), animated=True)
ims.savefig('./result/test_img.jpg')

# Plot the training losses.
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.savefig("Loss_Curve")