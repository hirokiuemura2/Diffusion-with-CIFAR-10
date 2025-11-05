import torch
from customDiffusion import UNet
import matplotlib.pyplot as plt
from imageFunctions import addNoise, betaSchedule, transform, train_loader, unnormalize, removeNoise
import torchvision.transforms as Tr
import torchvision

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net  = UNet().to(device)
    net.load_state_dict(torch.load('model_weights2.pth', weights_only=True))
    net.eval()

    T = 1000

    x, y = next(iter(train_loader))
    x = x[:8]  # Only using the first 8 for easy plotting
    y = y[:8]
    x = x.to(device)
    y = y.to(device)
    print(x.min(), x.max(), x.mean(), x.std())
    # Corrupt with a range of amounts
    t = torch.randint(30,T,(x.size(0),), device=x.device)
    noised_x, noise = addNoise(x, t, betaSchedule)
    newImages = noised_x.clone()
    # Get the model predictions
    
    with torch.no_grad():
        for i in range(5):
            preds = net(newImages, torch.tensor([1], device = device), T, y)
            newImages = removeNoise(newImages, t - i, preds, betaSchedule)
            print(newImages.min(), newImages.max(), newImages.mean(), newImages.std())
    preds = newImages.detach()
    # Plot fix colors!
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title("Input data")
    grid_in = torchvision.utils.make_grid(unnormalize(x).cpu(), nrow=8, normalize=False)
    axs[0].imshow(grid_in.permute(1,2,0))
    axs[1].set_title("Corrupted data")
    grid_no = torchvision.utils.make_grid(unnormalize(noised_x).cpu(), nrow=8, normalize=False)
    axs[1].imshow(grid_no.permute(1,2,0))
    axs[2].set_title("Network Predictions (denoised)")
    grid_pr = torchvision.utils.make_grid(unnormalize(preds).cpu(), nrow=8, normalize=False)
    axs[2].imshow(grid_pr.permute(1,2,0))
    fig.savefig("img3.png")

if __name__ == "__main__":
    main()