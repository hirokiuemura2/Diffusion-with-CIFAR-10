import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10 as CIFAR10
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

transform = T.Compose([
    T.Resize((32,32)),    # optional, sizes should already match 32X32
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range normalization
])


T = 200
betaSchedule = torch.linspace(0.0001, 0.02, T)

def addNoise(x0, t, betas):
    alphas = (1.0 - betas).to(device)
    alphas_cumulProd = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumulProd[t].view(-1,1,1,1).to(device)
    noise = torch.randn_like(x0)

    x_t = (torch.sqrt(a_t) * x0 + torch.sqrt(1-a_t) * noise)

    return x_t,noise

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()  # call the parent constructor

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(64, 64, kernel_size=5, padding=2),
                nn.Conv2d(64, 32, kernel_size=5, padding=2),
                nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x += h.pop()  # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x

batch_size = 64
n_epochs = 2

train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

net = UNet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []


for epoch in range(n_epochs):

    for x, y in train_loader:
        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        y = y.to(device)
        t = torch.randint(0, T, (x.size(0),), device=x.device)
        # noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x, noise = addNoise(x, t, betaSchedule)  # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x)

        # Calculate the loss
        loss = loss_fn(pred, noise)  # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_loader) :]) / len(train_loader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")
    
    # Fetch some data
x, y = next(iter(train_loader))
x = x[:8]  # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
t = torch.randint(0,T,(x.size(0),), device=x.device)
noised_x = addNoise(x, t, betaSchedule)

# Get the model predictions
with torch.no_grad():
    preds = net(noised_x.to(device)).detach().cpu()

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title("Input data")
axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap="Greys")
axs[1].set_title("Corrupted data")
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap="Greys")
axs[2].set_title("Network Predictions")
axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys")