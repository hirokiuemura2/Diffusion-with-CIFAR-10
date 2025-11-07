import torch
from customDiffusion import UNet
import matplotlib.pyplot as plt
from imageFunctions import addNoise, betaSchedule, transform, train_loader, unnormalize, removeNoise
import torchvision.transforms as Tr
import torchvision

def main():
    channel_size = 16 # Change Channel Count Here
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net  = UNet(mid_channels=channel_size).to(device) 
    net.load_state_dict(torch.load(f'model_weights_{channel_size}Channels.pth', weights_only=True, map_location=device)) # Change model_weights filename if needed
    net.eval()

    T = 1000
    sample = torch.randn(1, 3, 32, 32).to(device)
    
    with torch.no_grad():
        for i in range(T, 0, -1):
            print(i)
            t = torch.tensor([T - i], device=device)
            prompt = torch.tensor([2], device=device)  # Example prompt
            output = net(sample, t, T, prompt)
            sample = removeNoise(sample, t, output, betaSchedule).to(device)
            # print(sample.min(), sample.max(), sample.mean(), sample.std())
        # print(output.shape)
        # print(output)
        # print(output.min(), output.max())
        # print(output.mean(), output.std())
        # if preds.shape != sample.shape:
        #     if preds.ndim == 4 and preds.shape[0] == sample.shape[0] and preds.shape[-1] == sample.shape[1]:
        #         preds = preds.permute(0, 3, 1, 2).contiguous()
    sample = sample.to("cpu")

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

if __name__ == "__main__":
    main()