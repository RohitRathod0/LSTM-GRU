import torch.nn as nn
import torch

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, hid=128, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, out)
        )
    def forward(self, x): return self.net(x)

class NextTokenHead(nn.Module):
    def __init__(self, in_dim, vocab):
        super().__init__()
        self.proj = nn.Linear(in_dim, vocab)
    def forward(self, x): return self.proj(x)
