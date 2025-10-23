import torch
import torch.nn.functional as F

def bce_ls(logits, labels, ls=0.05):
    y = labels.float()*(1-ls) + 0.5*ls
    return F.binary_cross_entropy_with_logits(logits, y)

def next_token_loss(next_logits, next_targets):
    return F.cross_entropy(next_logits, next_targets)
