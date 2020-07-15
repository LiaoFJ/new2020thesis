import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
# Directory containing the data.
root = os.path.abspath('./data/')

def get_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize(90),
        transforms.ToTensor()])

    # Create the dataset.
    dataset = dset.ImageFolder(root=root, transform=transform)
    # dataset = dset.CIFAR10(
    #     root = "./data/",
    #     transform=transform,
    #     download=True
    # )
    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['batch_size'],
        shuffle=True)

    return dataloader