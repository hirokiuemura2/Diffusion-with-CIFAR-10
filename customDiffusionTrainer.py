import torch
import torchvision.transforms as Tr
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
from customDiffusion import UNet
import time

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

transform = Tr.Compose([
    Tr.Resize((32,32)),    # optional, sizes should already match 32X32
    Tr.RandomHorizontalFlip(),
    Tr.RandomVerticalFlip(),
    Tr.ToTensor(),
    Tr.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range normalization
])

def unnormalize(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x

def addNoise(x0, t, betas):
    alphas = (1.0 - betas).to(device)
    alphas_cumulProd = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumulProd[t].view(-1,1,1,1).to(device)
    noise = torch.randn_like(x0)

    x_t = (torch.sqrt(a_t) * x0 + torch.sqrt(1-a_t) * noise)

    return x_t,noise

batch_size = 64
n_epochs = 15
T = 1000
betaSchedule = torch.linspace(0.0001, 0.02, T).to(device)
losses = []

train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def train():
    net = UNet().to(device)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)

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
            pred = net(noisy_x, t, T, y)

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
                pred = net(noisy_x, t, T, y)
                loss = loss_fn(pred, noise)
                testLoss += loss.item()
        
        avgLoss = testLoss / len(test_loader)
        print(f"Test Loss: {avgLoss:.4f}")

    torch.save(net.state_dict(), 'model_weights2.pth')


if __name__ == "__main__":
    train()