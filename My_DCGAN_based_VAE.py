import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import register

nf = 32  # number of filters
nz = 100  # z dimension
nc = 1  # number of channels


# Sample z from mu and log_var

def sampling_from_latent_params(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample


# Network architecture is based on the paper -
# Likelihood regret: an out-of-distribution detection score for variational auto-encoder

# DCGAN_based_VAE model with adjustments

# - Last Conv2d layer in decoder is replaced by ConvTranspose2d, with out_channels = 1
# -  padding = 1 is added to ConvTranspose2d layers to make the decoder output 1x32x32

nf = 32
nz = 100
nc = 1

import torch.nn as nn


class MyDCGANvae(nn.Module):
    def __init__(self):
        super(MyDCGANvae, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, kernel_size=4, stride=2),  # 1*32*32 --> 32*15*15
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 32*15*15 --> 64*6*6
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # 64*6*6 --> 128*2*2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 200, kernel_size=4, stride=1, padding=1),  # 128*2*2 --> 200*1*1 , padding = 1
            nn.Flatten()

        )

        self.enc_to_mu = nn.Linear(in_features=200, out_features=nz)
        self.enc_to_log_var = nn.Linear(in_features=200, out_features=nz)
        # self.fc = nn.Linear(in_features = nz, out_features = nz)

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(nz, 128, kernel_size=4, stride=1),  # 128*4*4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32x16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1x32x32
            nn.Flatten()
        )

    def sampling_from_latent_params(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = self.encoder(x)

        mu = self.enc_to_mu(x)
        log_var = self.enc_to_log_var(x)

        latent_z = self.sampling_from_latent_params(mu, log_var)

        x = F.relu(latent_z)
        x = x.view(-1, 100, 1, 1)
        x_ = self.decoder(x)
        x_ = torch.sigmoid(x_)

        return x_, mu, log_var




@register("My_DCGAN_based_VAE")
def get_MyDCGAN_VAE():
    return MyDCGANvae()