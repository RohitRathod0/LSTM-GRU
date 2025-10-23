import random
import numpy as np

def cutmix_static(x1, x2, alpha=0.5):
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2

def token_dropout(tokens, p=0.1, unk_id=0):
    return [t if random.random() > p else unk_id for t in tokens]
