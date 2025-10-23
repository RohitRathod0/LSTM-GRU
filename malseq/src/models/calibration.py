import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor([init_T])))

    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T

    def nll_criterion(self, logits, labels):
        scaled = self.forward(logits)
        return torch.nn.functional.binary_cross_entropy_with_logits(scaled, labels.float())
