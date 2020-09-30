import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Loss function (Lvae) = Reconstruction Loss + KL Divergence
def Lvae(recon_x, x, mu, log_var, dim):
    reconstructionLoss = F.binary_cross_entropy(recon_x.view(-1, dim), x.view(-1, dim), reduction='sum')
    kl_Divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstructionLoss + kl_Divergence


# Training code based on - https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=PKNoWhdNYHB_

def train_model(c, model, train_loader, lr, epochs, dim=32):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if c == 1:
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, _) in enumerate(train_loader):
                data = data.reshape(-1, 1, dim, dim)
                optimizer.zero_grad()
                reconstruction, mu, log_var = model(data)
                loss = Lvae(reconstruction, data, mu, log_var, dim)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
        print('###### Epoch: {} Average loss: {:.4f}'.format(epoch, running_loss / len(train_loader.dataset)))

    elif c == 0:
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, _) in enumerate(train_loader):
                data = data.reshape(-1, dim * dim)
                optimizer.zero_grad()
                reconstruction, mu, log_var = model(data)
                loss = Lvae(reconstruction, data, mu, log_var, dim)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

            print('###### Epoch: {} Average loss: {:.4f}'.format(epoch, running_loss / len(train_loader.dataset)))


    # Visualize Results

    plt.figure(figsize=(10, 2))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(data[i].reshape(32, 32), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(model(data)[0].data[i].numpy().reshape(32, 32), cmap='gray')
        plt.axis('off')
        plt.savefig('Results.png')
