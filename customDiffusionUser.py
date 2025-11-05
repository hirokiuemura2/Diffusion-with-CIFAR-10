import torch
from customDiffusion import UNet
import matplotlib.pyplot as plt
from customDiffusionTrainer import transform, addNoise, betaSchedule, train_loader
import torchvision.transforms as Tr
import torchvision

def _unnormalize(x):
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    return x

def main():

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net  = UNet().to(device)
    net.load_state_dict(torch.load('model_weights2.pth', weights_only=True))
    net.eval()

    sample = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        t = torch.tensor([0], device=device)
        prompt = torch.tensor([2], device=device)  # Example prompt
        T = 1000  # Total time steps
        output = net(sample, t, T, prompt)
        print(output.shape)
        print(output)
        print(output.min(), output.max())
        print(output.mean(), output.std())

    # Visualize the input
    input_img = sample.squeeze(0).cpu()
    input_img = (input_img + 1) / 2  # Unnormalize to [0, 1]
    input_img = input_img.clamp(0, 1)
    input_img = input_img.permute(1, 2, 0).numpy()  # Change to HWC for plotting
    plt.imshow(input_img)
    plt.axis('off')
    plt.savefig("img5.png")

    # Visualize the output
    output = output.squeeze(0).cpu()
    output = (output + 1) / 2  # Unnormalize to [0, 1]
    output = output.clamp(0, 1)
    output = output.permute(1, 2, 0).numpy()  # Change to HWC for plotting
    plt.imshow(output)
    plt.axis('off')
    plt.savefig("img6.png")

    figure = plt.figure()



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
        preds = net(noised_x, t, T, y).detach()

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

if __name__ == "__main__":
    main()