import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
from dataset import DatasetFromFolder
from PIL import Image

# Directory containing the data.
root = os.path.abspath('./data_2/')
path = os.path.abspath('/')
def get_data(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

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

def get_data_for_test(params):
    """
    Loads the dataset and applies proproccesing steps to it.
    Returns a PyTorch DataLoader.
    """
    # Data proprecessing.
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


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
    print(dataloader)
    return dataloader

def get_train_data_set(path, params):
    dataset = DatasetFromFolder(path)
    dataloader = torch.utils.data.DataLoader(dataset,
        batch_size=params['batch_size'],
        shuffle=True)
    return dataloader

def get_test_img_single():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_img = Image.open('./test_image/test.jpg').convert("RGB")
    img = transform(test_img)
    img.unsqueeze_(0)
    return img