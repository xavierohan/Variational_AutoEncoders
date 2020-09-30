import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Simply_FC_VAE import sfcVAE
from DCGAN_based_VAE import DCGANvae
from train import train_model, Lvae
import argparse
import matplotlib.pyplot as plt

# Command Line Arguments

parser = argparse.ArgumentParser(description='Simply_FC_VAE , DCGAN_based_VAE ')
parser.add_argument('-d', '--data', type=int, required=True, help='0: MNIST, 1: FashionMNIST', default=0)
parser.add_argument('-m', '--model', type=int, required=True, help='0: Simply_FC_VAE, 1: DCGAN_based_VAE', default=0)
parser.add_argument('-t', '--train', type=int, required=True, help='0: Load pre-trained model, 1: train', default=1)
parser.add_argument('-r', '--lr', type=float, required=True, help='Enter lr', default=0.001)
parser.add_argument('-e', '--epochs', type=int, required=True, help='Enter epochs', default=1)
parser.add_argument('-z', '--zdim', type=int, required=True,
                    help='Enter latent dim for sfcVAE, 100 for DCGAN VAE', default=32)

args = vars(parser.parse_args())

# Resize all images to 32*32 as a standard for both models
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

if args["data"] == 1:
    train_data = datasets.FashionMNIST(root='../input/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
elif args["data"] == 0:
    train_data = datasets.MNIST(root='../input/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

z_dim = args["zdim"] # latent z dimension

if args["model"] == 0:
    for i, (data, _) in enumerate(train_loader):
        data = data.reshape(-1, 32 * 32)
    model = sfcVAE(z_dim)
    c = 0
elif args["model"] == 1:
    for i, (data, _) in enumerate(train_loader):
        data = data.reshape(-1, 1, 32,  32)
    model = DCGANvae()  # z_dim is fixed to 100
    c = 1
# c acts as a flag to help in data pre-processing in train_model

if args["train"] == 1:
    # Train model with passed arguments

    print("### TRAINING MODEL (Find results at Results.png) ###")
    train_model(c, model, train_loader, lr=args["lr"], epochs=args["epochs"])

elif args["train"] == 0:
    # Load Pre-trained model

    if args["model"] == 1:
        print("Loading Pre-trained DCGAV VAE, Find results at Results.png")
        if args["data"] == 1:
            model = torch.load('DCGAN_VAE_pretrained(FMNIST).pt')
            model.eval()
        elif args["data"] == 0:
            model = torch.load('DCGAN_VAE_pretrained(MNIST).pt')
            model.eval()

    elif args["model"] == 0:
        print("Loading Pre-trained Simply FC VAE, Find results at Results.png")
        if args["data"] == 1:
            model = torch.load('Simply_FC_VAE_pretrained(FMNIST).pt')
            model.eval()
        elif args["model"] == 0:
            model = torch.load('Simply_FC_VAE_pretrained(MNIST).pt')
            model.eval()

    # Visualize Training and Reconstructed images

    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(data[i].reshape(32, 32), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(model(data)[0].data[i].numpy().reshape(32, 32), cmap='gray')
        plt.axis('off')
        plt.savefig('Results.png')
