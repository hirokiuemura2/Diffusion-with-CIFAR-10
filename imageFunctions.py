import torch
import torchvision.transforms as Tr
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
T = 1000
batch_size = 192

transform = Tr.Compose([
    Tr.Resize((32,32)),    # optional, sizes should already match 32X32
    Tr.RandomHorizontalFlip(),
    Tr.RandomVerticalFlip(),
    Tr.ToTensor(),
    Tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range normalization
])
betaSchedule = torch.linspace(0.0001, 0.02, T).to(device)
train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def unnormalize(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x

def addNoise(x0, t, betas = betaSchedule):
    alphas = (1.0 - betas).to(device)
    alphas_cumulProd = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumulProd[t].view(-1,1,1,1).to(device)
    noise = torch.randn_like(x0)

    x_t = (torch.sqrt(a_t) * x0 + torch.sqrt(1-a_t) * noise)

    return x_t,noise

def removeNoise(x_t, t, noise, betas = betaSchedule):
    a_t = torch.cumprod(1.0 - betas, dim=0).to(device)[t].view(-1,1,1,1)

    sqrt_one_minus_a_t = torch.sqrt(1 - a_t)
    sqrt_a_t = torch.sqrt(a_t)

    x0_pred = ((x_t - sqrt_one_minus_a_t * noise) / sqrt_a_t).clamp(-1, 1)

    return x0_pred