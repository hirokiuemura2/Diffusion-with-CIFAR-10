import time
import torchvision
import torchvision.transforms as Tr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10 as CIFAR10
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import math

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

transform = Tr.Compose([
    Tr.Resize((32,32)),    # optional, sizes should already match 32X32
    Tr.RandomHorizontalFlip(),
    Tr.RandomVerticalFlip(),
    Tr.ToTensor(),
    Tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range normalization
])


T = 400
betaSchedule = torch.linspace(0.0001, 0.02, T).to(device)

def addNoise(x0, t, betas):
    alphas = (1.0 - betas).to(device)
    alphas_cumulProd = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumulProd[t].view(-1,1,1,1).to(device)
    noise = torch.randn_like(x0)

    x_t = (torch.sqrt(a_t) * x0 + torch.sqrt(1-a_t) * noise)

    return x_t,noise

def sinEmb(t, dim):
    device = t.device
    t = t.float() / float(T)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:,None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)
    return emb

def _unnormalize(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels = 64):
        super().__init__()  # call the parent constructor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb = nn.Sequential(
            nn.Linear(2 * mid_channels, 8 * mid_channels),
            nn.SiLU(),
            nn.Linear(8 * mid_channels, 2 * mid_channels)
        )
        self.prompt_emb = nn.Embedding(10, mid_channels * 2)

        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels, mid_channels * 2, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=5, padding=2),
            ]
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels * 4, mid_channels * 2, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, t, prompt):
        t_emb = sinEmb(t, self.down_layers[0].out_channels * 2)
        t_emb = self.time_emb(t_emb)
        p_emb = self.prompt_emb(prompt)
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i == 1:
                x = x + t_emb[:,:,None,None] + p_emb[:,:,None,None]   #time embedding layer
            if i < 2:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer
        x = self.dropout(x)

        for i, l in enumerate(self.up_layers):
            if i in (1,2):  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x = torch.cat([x, h.pop()], dim=1) # Fetching stored output (skip connection)
            x = self.act(l(x))  # Through the layer and the activation function

        return x
    
batch_size = 64
n_epochs = 5

train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

net = UNet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=5e-4)
losses = []


for epoch in range(n_epochs):
    net.train()
    startTime = time.time()
    for batch, (x, y) in enumerate(train_loader):
        size = len(train_loader.dataset)

        # Get some data and prepare the corrupted version
        x = x.to(device)  # Data on the GPU
        y = y.to(device)
        t = torch.randint(0, T, (x.size(0),), device=x.device)
        # noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
        noisy_x, noise = addNoise(x, t, betaSchedule)  # Create our noisy x

        # Get the model prediction
        pred = net(noisy_x, t, y)

        # Calculate the loss
        loss = loss_fn(pred, noise)  # How close is the output to the true 'clean' x?

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Store the loss for later
        losses.append(loss.item())
        if batch % 200 == 0:
            with torch.no_grad():
                p = pred.detach()
                # print basic stats to spot collapse or explode
                print(f"batch {batch} loss {loss.item():.4f} pred mean {p.mean().item():.4f} std {p.std().item():.4f} x mean {x.mean().item():.4f} std {x.std().item():.4f}")

    # Print our the average of the loss values for this epoch:
    endTime = time.time()
    print(f"Epoch took {endTime - startTime:.2f} seconds")
    avg_loss = sum(losses[-len(train_loader) :]) / len(train_loader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

    net.eval()
    testLoss = 0.0
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, T, (x.size(0),), device=x.device)
            noisy_x, noise = addNoise(x, t, betaSchedule)  # Create our noisy x
            pred = net(noisy_x, t, y)
            loss = loss_fn(pred, noise)
            testLoss += loss.item()
    
    avgLoss = testLoss / len(test_loader)
    print(f"Test Loss: {avgLoss:.4f}")

# Fetch some data
x, y = next(iter(train_loader))
x = x[:8]  # Only using the first 8 for easy plotting
y = y[:8]
x = x.to(device)
y = y.to(device)

# Corrupt with a range of amounts
t = torch.randint(0,T,(x.size(0),), device=x.device)
noised_x, noise = addNoise(x, t, betaSchedule)

# Get the model predictions
alphas = (1.0 - betaSchedule).to(device)
alpha_bars = torch.cumprod(alphas, dim=0).to(device)
with torch.no_grad():
    preds = net(noised_x, t, y).detach()

    # If network returned channels-last (B,H,W,C) try to convert to (B,C,H,W)
    if preds.shape != noised_x.shape:
        if preds.ndim == 4 and preds.shape[0] == noised_x.shape[0] and preds.shape[-1] == noised_x.shape[1]:
            preds = preds.permute(0, 3, 1, 2).contiguous()

    # build per-sample alpha tensors with explicit shape (B,1,H,W)
    B, C, H, W = noised_x.shape
    alpha_bar_t = alpha_bars[t].to(noised_x.device).float().view(-1, 1, 1, 1).expand(B, 1, H, W)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    # compute predicted x0 (broadcastable shapes guaranteed)
    x0_pred = (noised_x - sqrt_one_minus_alpha_bar_t * preds) / (sqrt_alpha_bar + 1e-8)
    preds = x0_pred.cpu()

# # Plot fix colors!
fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].set_title("Input data")
grid_in = torchvision.utils.make_grid(_unnormalize(x).cpu(), nrow=8, normalize=False)
axs[0].imshow(grid_in.permute(1,2,0).clip(0,1))
axs[1].set_title("Corrupted data")
grid_no = torchvision.utils.make_grid(_unnormalize(noised_x).cpu(), nrow=8, normalize=False)
axs[1].imshow(grid_no.permute(1,2,0).clip(0,1))
axs[2].set_title("Network Predictions (denoised)")
grid_pr = torchvision.utils.make_grid(_unnormalize(preds).cpu(), nrow=8, normalize=False)
axs[2].imshow(grid_pr.permute(1,2,0).clip(0,1))
fig.savefig("img3.png")

sample = torch.randn(8,3,32,32).to(device)
y = torch.tensor([0,1,2,3,4,5,6,7], device = device)
alphas = (1.0 - betaSchedule).to(device)
alpha_bars = torch.cumprod(alphas, dim=0).to(device)

# sampling: replace for-loop with broadcasting-safe variant
sample = torch.randn(8,3,32,32).to(device)
y = torch.tensor([0,1,2,3,4,5,6,7], device = device)
alphas = (1.0 - betaSchedule).to(device)
alpha_bars = torch.cumprod(alphas, dim=0).to(device)

# for time_step in reversed(range(T)):
for time_step in range(T-1, T-2, -1):
    with torch.no_grad():
        t_batch = torch.full((sample.size(0),), time_step, device=sample.device, dtype=torch.long)
        preds = net(sample, t_batch, y).detach()

        # fix possible channel-order mismatch
        if preds.shape != sample.shape:
            if preds.ndim == 4 and preds.shape[0] == sample.shape[0] and preds.shape[-1] == sample.shape[1]:
                preds = preds.permute(0, 3, 1, 2).contiguous()

        B, C, H, W = sample.shape
        # scalar -> per-sample (B,1,H,W)
        alpha_t_scalar = alphas[time_step].to(sample.device).float()
        alpha_bar_scalar = alpha_bars[time_step].to(sample.device).float()
        alpha_t_b = alpha_t_scalar.view(1,1,1,1).expand(B, 1, H, W)
        alpha_bar_b = alpha_bar_scalar.view(1,1,1,1).expand(B, 1, H, W)

        sqrt_alpha_bar = torch.sqrt(alpha_bar_b)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_b)

        x0_pred = (sample - sqrt_one_minus_alpha_bar_t * preds) / (sqrt_alpha_bar + 1e-8)

        if time_step > 0:
            noise = torch.randn_like(sample)
        else:
            noise = torch.zeros_like(sample)
        sample = torch.sqrt(alpha_t_b) * x0_pred + torch.sqrt(1.0 - alpha_t_b) * noise
print(sample[0])
sample = sample.clamp(-1, 1)
sample_vis = _unnormalize(sample)
grid = torchvision.utils.make_grid(sample_vis.cpu(), nrow=8, normalize=False)
img = grid.permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(12, 7))
plt.imshow(img)
plt.axis("off")

labels = ["airplane", "automobile", "bird", "cat", "deer",
 "dog", "frog", "horse", "ship", "truck"]

for i, label in enumerate(y):
    plt.text(
        x=4 + i * (img.shape[1] / len(y)),  # rough spacing for nrow=8
        y=img.shape[0] - 30,                # near bottom
        s=labels[label.item()],
        color="white",
        fontsize=12,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.5, pad=2)
    )
plt.title("Generated Samples with Labels")
plt.savefig("img4.png")