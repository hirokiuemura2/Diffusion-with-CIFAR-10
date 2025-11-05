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

class FiLM(nn.Module):
    def __init__(self, in_channels, time_emb_dim):
        super().__init__()
        self.linear = nn.Linear(time_emb_dim, in_channels * 2)

    def forward(self, emb):
        # emb: [batch, in_dim]
        gamma_beta = self.mlp(emb)  # [batch, 2*out_dim]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return gamma, beta

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels = 64):
        super().__init__()  # call the parent constructor

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.time_emb = nn.Sequential(
            nn.Linear(2 * mid_channels, 8 * mid_channels),
            nn.SiLU(),
            nn.Linear(8 * mid_channels, 2 * mid_channels)
        )
        self.time_emb2 = nn.Sequential(
            nn.Linear(2 * mid_channels, 8 * mid_channels),
            nn.SiLU(),
            nn.Linear(8 * mid_channels, 4 * mid_channels)
        )
        self.prompt_emb = nn.Embedding(10, mid_channels * 2)
        self.prompt_emb2 = nn.Embedding(10, mid_channels * 4)

        self.down_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels * 2, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 2),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 2, mid_channels * 4, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 4),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 4, mid_channels * 6, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 6),
                    nn.SiLU()
                )
            ]
            
        )
        self.up_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(mid_channels * 6, mid_channels * 4, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 4),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 8, mid_channels * 4, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 4),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 6, mid_channels * 2, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 2),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 3, out_channels * 2, kernel_size=5, padding=2),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.SiLU()
                )
            ]
        )
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout = nn.Dropout2d(0.1)
        self.finalLayer = nn.Conv2d(out_channels * 2, out_channels, kernel_size=5, padding=2)
    
    def forward(self, x, t, T, prompt):
        # use stored mid_channels instead of accessing attributes on Sequential
        t_emb = sinEmb(t, T, self.mid_channels * 2)
        t_emb = self.time_emb(t_emb)
        t_emb2 = self.time_emb2(t_emb)
        p_emb = self.prompt_emb(prompt)
        p_emb2 = self.prompt_emb2(prompt)
        h = []
        for i, l in enumerate(self.down_layers):
            x = l(x)  # Through the layer and the activation function
            if i == 1:
                x = x + t_emb[:,:,None,None] + p_emb[:,:,None,None]   #time embedding layer
            elif i == 2:
                x = x + t_emb2[:,:,None,None] + p_emb2[:,:,None,None] # second time embedding layer
            if i < 3:  # For all but the third (final) down layer:
                h.append(x)  # Storing output for skip connection
                x = self.downscale(x)  # Downscale ready for the next layer
        x = self.dropout(x)

        for i, l in enumerate(self.up_layers):
            if i > 0:  # For all except the first up layer
                x = self.upscale(x)  # Upscale
                x = torch.cat([x, h.pop()], dim=1) # Fetching stored output (skip connection)
            x = l(x)
        x = self.finalLayer(x)
        return x