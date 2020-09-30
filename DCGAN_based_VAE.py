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

class DCGANvae(nn.Module):
    def __init__(self):
        super(DCGANvae, self).__init__()

        # ENCODER
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
        # self.fc = nn.Linear(in_features=nz, out_features=nz)

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nz, 128, kernel_size=4, stride=1),  # 128*4*4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),  # 64*10*10
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),  # 32*22*22
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=4, stride=2),  # 256*10*10
            nn.Flatten()
        )

        self.fc2 = nn.Linear(in_features=256 * 10 * 10, out_features=32 * 32)  #

    def forward(self, x):
        x = self.encoder(x)

        mu = self.enc_to_mu(x)
        log_var = self.enc_to_log_var(x)

        latent_z = sampling_from_latent_params(mu, log_var)

        # x = F.relu(self.fc(latent_z))
        x = F.relu(latent_z)
        x = x.view(-1, nz, 1, 1)
        x_ = self.decoder(x)
        x_ = torch.sigmoid(self.fc2(x_))  #

        return x_, mu, log_var


@register("DCGAN_based")
def get_DCGAN_VAE():
    return DCGANvae()
