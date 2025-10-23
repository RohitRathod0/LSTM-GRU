import torch
import torch.nn as nn

class FusionGate(nn.Module):
    def __init__(self, in_dim, hid=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1),
            nn.Sigmoid()
        )

    def forward(self, s_vec):
        return self.net(s_vec)  # [B,1]
