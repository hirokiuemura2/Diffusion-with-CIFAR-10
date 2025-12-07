import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class NeuralNet(nn.Module):
    def __init__(self, img_size, timestep_count):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * img_size * img_size, 128),
            nn.SiLU(),
            nn.Linear(128, timestep_count)
        )

    def forward(self, x):
        x = self.layers(x)
        return x