import torch
from customDiffusion import UNet
import matplotlib.pyplot as plt
from imageFunctions import addNoise, betaSchedule, transform, train_loader, unnormalize, removeNoise
import torchvision.transforms as Tr
import torchvision

def main():
    channel_size = 32 # Change Channel Count Here
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net  = UNet(mid_channels=channel_size).to(device) 
    net.load_state_dict(torch.load(f"model_weights{channel_size}Channels.pth", weights_only=True, map_location=device)) # Change model_weights filename if needed
    net.eval()

    T = 1000

    x, y = next(iter(train_loader))
    x = x[:1]  # Only using the first 1 for easy plotting
    y = y[:1]
    x = x.to(device)
    y = y.to(device)
    # print(x.min(), x.max(), x.mean(), x.std())
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 
               10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 45, 
               50, 60, 70, 80, 90, 
               100, 150, 200, 300, 400, 500, 700, 1000]
    
    t = torch.tensor(indices, device=x.device).long()
    
    x0_batch = x[0].unsqueeze(0).repeat(t.size(0), 1, 1, 1)  # Create a batch of the same image
    y0_batch = y[0].unsqueeze(0).repeat(t.size(0))  # Create a batch of the same label
    finalImages = x[0].unsqueeze(0).repeat(t.size(0) + 2, 1, 1, 1)
    x0_batch, noise = addNoise(x0_batch, t, betaSchedule)
    finalImages = x[0].unsqueeze(0).repeat(t.size(0) + 2, 1, 1, 1)
    finalImages[1] = x0_batch[0].clone().detach()
    with torch.no_grad():
        for i in range(1, T):
            preds = net(x0_batch, t, T, y0_batch)
            x0_batch = removeNoise(x0_batch, t, preds, betaSchedule)
            if (indices[i] == i):
                finalImages[i + 2] = x0_batch[i].clone().detach()
    
    input_img = x[0].squeeze(0).cpu()
    input_img = (input_img + 1) / 2  # Unnormalize to [0, 1]
    # input_img = input_img.clamp(0, 1)
    input_img = input_img.permute(1, 2, 0).numpy()  # Change to HWC for plotting
    plt.imshow(input_img)
    plt.axis('off')
    plt.savefig("img1.png")
    
    input_img = x0_batch[1].squeeze(0).cpu()
    input_img = (input_img + 1) / 2  # Unnormalize to [0, 1]
    # input_img = input_img.clamp(0, 1)
    input_img = input_img.permute(1, 2, 0).numpy()  # Change to HWC for plotting
    plt.imshow(input_img)
    plt.axis('off')
    plt.savefig("img2.png")
    

#     fig, axs = plt.subplots(denoiseCount + 2, 1, figsize=(12, 7))
#     axs[0].set_title("Input data")
#     grid_in = torchvision.utils.make_grid(unnormalize(x).cpu(), nrow=8, normalize=False)
#     axs[0].imshow(grid_in.permute(1,2,0))
#     axs[1].set_title("Corrupted data")
#     grid_no = torchvision.utils.make_grid(unnormalize(noised_x).cpu(), nrow=8, normalize=False)
#     axs[1].imshow(grid_no.permute(1,2,0))


#     # Get the model predictions
#     cols = 10
#     rows = 10
#     figure = plt.figure(figsize=(15, 15))
    
#     for i in range(0, cols * rows):
#         img = x0_norm[i+100]
#         label = f"t = {i}"
#         figure.add_subplot(rows, cols, i + 1)
#         plt.title(label)
#         plt.axis("off")
#         plt.imshow(img.permute(1,2,0).squeeze(), cmap="gray")
#     plt.savefig("plot3.png") #plt.show()
    
#     with torch.no_grad():
#         for i in range(denoiseCount):
#             preds = net(newImages, torch.tensor(t, device = device), T, y)
#             newImages = removeNoise(newImages, t, preds, betaSchedule)
#             t = t - 1
#             print(newImages.min(), newImages.max(), newImages.mean(), newImages.std())

#             preds = newImages.clone().detach()
#             grid_pr = torchvision.utils.make_grid(unnormalize(preds).cpu(), nrow=8, normalize=False)
#             axs[2 + i].imshow(grid_pr.permute(1,2,0))

    
#     fig.savefig("img3.png")

if __name__ == "__main__":
    main()