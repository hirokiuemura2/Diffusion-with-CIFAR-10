import torchvision
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10
import random

import matplotlib.pyplot as plt


transform = T.Compose([
    T.Resize((32,32)),    # optional, sizes should already match 32X32
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range normalization
])

transform_basic = T.ToTensor()

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

indx = random.randint(0,len(train_data)-1)
imgTensor, label = train_data[indx]
print(label)
img = imgTensor.permute(1,2,0).numpy()
print(img)
plt.imshow(img)
plt.title(f"Label: {train_data.classes[label]}")
plt.axis('off')
plt.savefig('img.png')

T = 200
betaSchedule = torch.linspace(0.0001, 0.02, T)

def addNoise(x0, t, betas):
    alphas = 1.0 - betas
    alphas_cumulProd = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumulProd[t].view(-1,1,1,1)
    noise = torch.randn_like(x0)

    x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1-a_t) * noise

    return x_t,noise

figure = plt.figure(figsize=(15, 15))
cols, rows = 10,10

x0_batch = imgTensor.unsqueeze(0).repeat(200, 1, 1, 1)

t = torch.linspace(0, 199,200).long()

x0_batch,noise = addNoise(x0_batch,t,betaSchedule)

x_min = x0_batch.view(x0_batch.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
x_max = x0_batch.view(x0_batch.size(0), -1).max(dim=1)[0].view(-1,1,1,1)

x0_norm = (x0_batch - x_min) / (x_max - x_min)


print(noise.shape)

for i in range(1, cols * rows + 1):
    img = x0_norm[i]
    label = f"t = {i}"
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.permute(1,2,0).squeeze(), cmap="gray")
plt.savefig("plot3.png") #plt.show()