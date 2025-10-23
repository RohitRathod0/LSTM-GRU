import numpy as np
from scipy.stats import entropy

def population_stability_index(ref_hist, cur_hist, eps=1e-8):
    ref = np.asarray(ref_hist) + eps
    cur = np.asarray(cur_hist) + eps
    ref /= ref.sum()
    cur /= cur.sum()
    return ((cur - ref) * np.log(cur/ref)).sum()

def token_kl(ref_counts, cur_counts, eps=1e-8):
    ref = np.asarray(ref_counts) + eps
    cur = np.asarray(cur_counts) + eps
    ref /= ref.sum()
    cur /= cur.sum()
    return float(entropy(cur, ref))
