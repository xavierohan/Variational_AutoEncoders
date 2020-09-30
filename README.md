# Variational_AutoEncoders

## Getting Started

### This repository contains the following Variational AutoEncoder (VAE) Models:

-- Simply Fully Connected VAE

-- DCGAN based VAE

### These models are trained on the following datasets:

-- MNIST

-- FashionMNIST

### Pre-trained Models can be downloaded from - [Pre-trained Models](https://drive.google.com/drive/folders/1Nk3xpGvYcnHxkO7p8PRHpRmUvVhFjeDV?)

## To Run From Your Terminal

### Arguments to be passed:

```
d - Dataset - ( 0 : MNIST | 1 : FashionMNIST )
m - model - ( 0 : Simply_FC_VAE | 1 : DCGAN_based_VAE )
t - train - ( 0 : Load Pre-trained Model | 1 : Train Model )
r - learning rate - ( Enter Learning Rate (lr) )
e - epochs - ( Enter number of iterations )
z - z dim - ( Enter Latent z dimension, z is set to 100 for DCGAN_based_VAE )
```
### Example:

To train the Simply_FC_VAE model on the MNIST dataset with number of epochs = 20, learning rate = 0.001 and latent dimension = 16
```
python3 main.py -d 0 -m 0 -t 1 -r 0.001 -e 20 -z 16
```
