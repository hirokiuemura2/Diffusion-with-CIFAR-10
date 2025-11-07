import torch
import torchvision.transforms as Tr
import torch.nn as nn
import time
from customDiffusionShallow import UNet
from imageFunctions import transform, addNoise, betaSchedule, T, train_loader, test_loader

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
n_epochs = 200

losses = []
channelSize = 24 # Change Channel Count Here

net = UNet(mid_channels=channelSize).to(device)
print(sum([p.numel() for p in net.parameters()]), "parameters")
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    if epoch == 10:
        opt = torch.optim.Adam(net.parameters(), lr=4e-5)
    if epoch == 50: opt = torch.optim.Adam(net.parameters(), lr=2e-6)
    if epoch == 100: opt = torch.optim.Adam(net.parameters(), lr=1e-7)
    net.train()
    startTime = time.time()
    size = len(train_loader.dataset)
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)  # Data on the GPU
        y = y.to(device)
        t = torch.randint(0, T, (x.size(0),), device=x.device) # random timesteps
        noisy_x, noise = addNoise(x, t, betaSchedule)  # noisy image + noise
        pred = net(noisy_x, t, T, y) #net prediction

        # Calculate the loss
        loss = loss_fn(pred, noise)  # noise is target

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        # Store the loss for later
        losses.append(loss.item())
        # if batch % 200 == 0:
        #     with torch.no_grad():
        #         p = pred.detach()
                # print basic stats to spot collapse or explode
                # print(f"batch {batch} loss {loss.item():.4f} pred mean {p.mean().item():.4f} std {p.std().item():.4f} x mean {x.mean().item():.4f} std {x.std().item():.4f}")

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

torch.save(net.state_dict(), f'model_weights_{channelSize}Channels.pth')