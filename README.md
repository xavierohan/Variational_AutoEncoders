# Variational_AutoEncoders

## Getting Started

### This repository contains the following Variational AutoEncoder (VAE) Models:

-- Simply Fully Connected VAE

-- DCGAN based VAE ( Based on "Likelihood regret: an out-of-distribution detection score for variational auto-encoder", https://arxiv.org/pdf/2003.02977.pdf)

-- MyDCGAN based VAE ( Based on the above model, but with slight alterations )
```
    - Final conv2d layer in decoder is replaced by ConvTranspose2d  
    - Padding = 1 added to all but first ConvTranspose2d layer in the decoder
    - out_channels of final layer = 1, to get 1x32x32 output
```
    

### These models are trained on the following datasets:

-- MNIST

-- FashionMNIST

### Pre-trained Models (for Simply Fully Connected VAE and DCGAN based VAE) can be downloaded from ( Add them to the folder containing main.py ) - [Pre-trained Models](https://drive.google.com/drive/folders/1Nk3xpGvYcnHxkO7p8PRHpRmUvVhFjeDV?)
P.S. The pre-trained models are only trained upto a few epochs. 

## To Run From Your Terminal

### Arguments to be passed:

```
d - Dataset - ( 0 : MNIST | 1 : FashionMNIST )
m - model - ( 0 : Simply_FC_VAE | 1 : DCGAN_based_VAE  | 2: MyDCGAN_based_VAE)
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
Reconstruction results will be saved under Results.png

<img src="https://github.com/xavierohan/Variational_AutoEncoders/blob/master/Results.png" width="400">

First Row - Original image

Second Row - Reconstructed image
