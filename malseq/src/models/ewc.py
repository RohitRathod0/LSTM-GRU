import torch
from torch import nn
from collections import defaultdict

class EWCWrapper:
    def __init__(self, model: nn.Module, lambda_=50.0):
        self.model = model
        self.lambda_ = lambda_
        self.fisher = defaultdict(lambda: 0.0)
        self.opt_par = {}

    @torch.no_grad()
    def compute_fisher(self, data_loader, loss_fn, device):
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        for batch in data_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = self.model(batch)
            loss = loss_fn(out["logit"], batch["label"])
            self.model.zero_grad()
            loss.backward()
            for (n, p) in self.model.named_parameters():
                if p.grad is not None and p.requires_grad:
                    fisher[n] += p.grad.detach() ** 2
        for n in fisher:
            fisher[n] /= len(data_loader)
        self.fisher = fisher
        self.opt_par = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                loss += (self.fisher[n] * (p - self.opt_par[n])**2).sum()
        return self.lambda_ * loss
