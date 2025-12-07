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
        self.linear = nn.Linear(in_channels, time_emb_dim * 2)

    def forward(self, emb):
        # emb: [batch, in_dim]
        gamma_beta = self.linear(emb)  # [batch, 2*out_dim]
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
        self.prompt_emb = nn.Embedding(10, mid_channels * 2)

        self.time_filmUp = nn.ModuleList(
            [
                FiLM(2 * mid_channels, mid_channels),
                FiLM(2 * mid_channels, 2 * mid_channels),
                FiLM(2 * mid_channels, 4 * mid_channels),
                FiLM(2 * mid_channels, 4 * mid_channels),
                FiLM(2 * mid_channels, 6 * mid_channels),
                FiLM(2 * mid_channels, 4 * mid_channels),
                FiLM(2 * mid_channels, 4 * mid_channels),
                FiLM(2 * mid_channels, 2 * mid_channels),
                FiLM(2 * mid_channels, mid_channels),
            ]
        )

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
                    nn.Conv2d(mid_channels * 4, mid_channels * 4, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 6),
                    nn.SiLU()
                ),
                nn.Sequential(
                    nn.Conv2d(mid_channels * 4, mid_channels * 6, kernel_size=5, padding=2),
                    nn.BatchNorm2d(mid_channels * 4),
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
                    nn.BatchNorm2d(mid_channels * 2),
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
                    nn.Conv2d(mid_channels * 3, mid_channels, kernel_size=5, padding=2),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.SiLU()
                )
            ]
        )
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout = nn.Dropout2d(0.15)
        self.finalLayer = nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias = True)
    
    def forward(self, x, t, T, prompt):
        t_emb = sinEmb(t, T, self.mid_channels * 2)
        t_emb = self.time_emb(t_emb)
        p_emb = self.prompt_emb(prompt)
        h = []  # For skip connections


        for i, l in enumerate(self.down_layers):
            x = l(x)  # Layer application, normalization, activation
            gamma, beta = self.time_filmUp[i](t_emb + p_emb)
            x = gamma[:, :, None, None] * x + beta[:, :, None, None] # FiLM conditioning
            if i < len(self.down_layers) - 1:  # For all but the last down layer:
                h.append(x)  # Skip connection storage
            if i > 0:  # Downscale after first layer
                x = self.downscale(x)
        x = self.dropout(x)

        for i, l in enumerate(self.up_layers):
            if i > 0 and i < len(self.up_layers - 1):
                    x = self.upscale(x)
            if i > 0:
                x = torch.cat([x, h.pop()], dim=1) # Skip
            x = l(x)
            if (i < len(self.up_layers) - 1):
                gamma, beta = self.time_filmUp[i + 5](t_emb + p_emb) #conditioning
                x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        x = self.finalLayer(x)
        return x