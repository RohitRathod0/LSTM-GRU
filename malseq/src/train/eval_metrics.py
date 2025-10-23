import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

def evaluate_probs(y_true, y_prob, thresh=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_prob),
    }
