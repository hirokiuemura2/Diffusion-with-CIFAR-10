import torch
from DDPM import UNet
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
net  = UNet().to(device)
net.load_state_dict(torch.load('model_weights.pth', weights_only=True))
net.eval()

sample = torch.randn(1, 3, 32, 32).to(device)
with torch.no_grad():
    t = torch.tensor([0], device=device)
    prompt = torch.tensor([0], device=device)  # Example prompt
    T = 400  # Total time steps
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