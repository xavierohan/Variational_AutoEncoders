# Simply fully connected VAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from registry import register

# Code based on https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
# Sample z using mu and log_var 

def sampling_from_latent_params(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample


class sfcVAE(nn.Module):
    def __init__(self, z_dim, dim_=32 * 32):
        super(sfcVAE, self).__init__()

        # ENCODER
        self.enc_1 = nn.Linear(in_features=dim_, out_features=512)
        self.enc_2 = nn.Linear(in_features=512, out_features=128)

        self.enc_to_mu = nn.Linear(in_features=128, out_features=z_dim)
        self.enc_to_log_var = nn.Linear(in_features=128, out_features=z_dim)

        # DECODER
        self.dec_1 = nn.Linear(in_features=z_dim, out_features=128)
        self.dec_2 = nn.Linear(in_features=128, out_features=512)
        self.dec_3 = nn.Linear(in_features=512, out_features=dim_)

    def forward(self, x):
        x = F.relu(self.enc_1(x))
        x = F.relu(self.enc_2(x))

        mu = self.enc_to_mu(x)
        log_var = self.enc_to_log_var(x)

        latent_z = sampling_from_latent_params(mu, log_var)

        x = F.relu(self.dec_1(latent_z))
        x = F.relu(self.dec_2(x))

        x_ = torch.sigmoid(self.dec_3(x))

        return x_, mu, log_var

@register("simply_FC")
def get_sfcvae():
    return sfcVAE()

