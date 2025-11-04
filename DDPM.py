import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def sinEmb(t, T, dim):
    device = t.device
    t = t.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(T) * torch.arange(0, half, device=t.device) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    return emb

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
                nn.Conv2d(mid_channels * 2, mid_channels * 2, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=5, padding=2),
                nn.Conv2d(mid_channels , out_channels, kernel_size=5, padding=2, bias = True),
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, t, T, prompt):
        t_emb = sinEmb(t, T, self.down_layers[0].out_channels * 2)
        t_emb = self.time_emb(t_emb)
        p_emb = self.prompt_emb(prompt)
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # Through the layer and the activation function
            if i == 1:
                x = x + t_emb[:,:,None,None] + p_emb[:,:,None,None]   #time embedding layer
            if i < 3:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer
        x = self.dropout(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                # x = torch.cat([x, h.pop()], dim=1) # Fetching stored output (skip connection)
            x = l(x)
            if i != len(self.up_layers) - 1:
                x = self.act(x)
            #try: normalize, activate, then convolute again? or: just normalize and convolute down to output?
        # print("Pre-tanh min/max:", x.min().item(), x.max().item())  # Debug line
        # x = nn.Linear(in_features=x.shape[1], out_features=self.out_channels)(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.up_layers[-1].weight)
        nn.init.constant_(self.up_layers[-1].bias, -0.5)