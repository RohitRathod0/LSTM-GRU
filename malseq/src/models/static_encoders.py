import torch
import torch.nn as nn

class StaticEncoder(nn.Module):
    def __init__(self, pe_dim, import_dim, hid=128):
        super().__init__()
        self.pe_mlp = nn.Sequential(
            nn.Linear(pe_dim, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.imp_mlp = nn.Sequential(
            nn.Linear(import_dim, hid),
            nn.ReLU(),
            nn.BatchNorm1d(hid),
            nn.Linear(hid, hid//2),
            nn.ReLU(),
        )
        self.out = nn.Linear(hid + hid//2, hid)

    def forward(self, pe_vec, import_vec):
        a = self.pe_mlp(pe_vec)
        b = self.imp_mlp(import_vec)
        x = torch.cat([a, b], dim=-1)
        return self.out(x)
