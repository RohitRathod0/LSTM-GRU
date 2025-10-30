import os, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# ============== THREE MODEL ARCHITECTURES ==============

class LSTM_Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, num_classes))
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class GRU_Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, num_classes))
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

class MLP_Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, width=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.ReLU(), nn.BatchNorm1d(width), nn.Dropout(dropout),
            nn.Linear(width, width//2), nn.ReLU(), nn.BatchNorm1d(width//2), nn.Dropout(dropout),
            nn.Linear(width//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ============== DATA LOADING ==============

def load_data(csv_path, label_col="label", test_size=0.15, val_size=0.10, seed=42, max_rows=None):
    df = pd.read_csv(csv_path, nrows=max_rows)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    y = df[label_col].values.astype(np.int64)
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=test_size+val_size, random_state=seed, stratify=y)
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=1-rel_val, random_state=seed, stratify=y_tmp)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    num_classes = len(np.unique(y))
    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, num_classes)

def batch_iter_mlp(X, y, batch=1024, shuffle=True):
    idx = np.arange(len(y))
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx), batch):
        j = idx[i:i+batch]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j]).long()

def batch_iter_seq(X, y, batch=1024, seq_len=20, shuffle=True):
    idx = np.arange(len(y))
    if shuffle: np.random.shuffle(idx)
    total_features = X.shape[1]
    features_per_step = total_features // seq_len
    truncated_features = features_per_step * seq_len
    X_truncated = X[:, :truncated_features]
    for i in range(0, len(idx), batch):
        j = idx[i:i+batch]
        xb = X_truncated[j].reshape(-1, seq_len, features_per_step)
        yb = y[j]
        yield torch.from_numpy(xb), torch.from_numpy(yb).long()

# ============== TRAINING ==============

def train(csv_path, model_type="lstm", outdir="runs/model", seed=42, epochs=10, batch=1024, lr=3e-4, wd=1e-4, max_rows=None, label_col="label", seq_len=20):
    os.makedirs(outdir, exist_ok=True)
    Xtr, ytr, Xva, yva, Xte, yte, scaler, num_classes = load_data(csv_path, label_col=label_col, max_rows=max_rows, seed=seed)
    total_features = Xtr.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}\nTraining {model_type.upper()} - Malware Detection\n{'='*60}")
    print(f"Train: {len(ytr):,}, Val: {len(yva):,}, Test: {len(yte):,}")
    print(f"Features: {total_features}, Classes: {num_classes}, Device: {device}\n{'='*60}\n")
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "lstm":
        features_per_step = total_features // seq_len
        model = LSTM_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        batch_fn = lambda X, y, b, s: batch_iter_seq(X, y, b, seq_len, s)
    elif model_type == "gru":
        features_per_step = total_features // seq_len
        model = GRU_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        batch_fn = lambda X, y, b, s: batch_iter_seq(X, y, b, seq_len, s)
    elif model_type == "mlp":
        model = MLP_Classifier(total_features, num_classes, width=256).to(device)
        batch_fn = batch_iter_mlp
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in batch_fn(Xtr, ytr, batch, True):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        model.eval()
        vs, ps = [], []
        with torch.no_grad():
            for xb, yb in batch_fn(Xva, yva, batch, False):
                xb = xb.to(device)
                prob = F.softmax(model(xb), dim=-1).cpu().numpy()
                ps.append(prob)
                vs.append(yb.numpy())
        ps = np.vstack(ps)
        vs = np.concatenate(vs)
        acc = (ps.argmax(1) == vs).mean()
        print(f"Epoch {ep:2d}/{epochs} | Loss {tr_loss:7.3f} | Val Acc {acc:.4f} ({acc*100:.2f}%)")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(outdir, "best.pt"))

    print(f"\n{'='*60}\nTesting on held-out test set...\n{'='*60}")
    model.load_state_dict(torch.load(os.path.join(outdir, "best.pt"), map_location=device))
    model.eval()
    vs, ps = [], []
    with torch.no_grad():
        for xb, yb in batch_fn(Xte, yte, batch, False):
            xb = xb.to(device)
            prob = F.softmax(model(xb), dim=-1).cpu().numpy()
            ps.append(prob)
            vs.append(yb.numpy())
    ps = np.vstack(ps)
    vs = np.concatenate(vs)
    test_acc = (ps.argmax(1) == vs).mean()
    y_pred = ps.argmax(1)
    print(f"\n{model_type.upper()} Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    if num_classes == 2:
        auc = roc_auc_score(vs, ps[:,1])
        print(f"ROC-AUC: {auc:.4f}\n")
    target_names = ['Normal', 'Attack'] if num_classes == 2 else [f'Class_{i}' for i in range(num_classes)]
    print(classification_report(vs, y_pred, target_names=target_names, digits=4, zero_division=0))
    metrics = {"model": model_type, "test_accuracy": float(test_acc), "val_accuracy": float(best_acc), "num_classes": num_classes}
    if num_classes == 2:
        metrics["roc_auc"] = float(auc)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved model and metrics to: {outdir}\n{'='*60}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", default="label")
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "mlp"])
    ap.add_argument("--outdir", default="runs/model")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=20)
    args = ap.parse_args()
    train(args.csv, model_type=args.model, outdir=args.outdir, epochs=args.epochs, batch=args.batch, lr=args.lr, wd=args.wd, max_rows=args.max_rows, label_col=args.label, seq_len=args.seq_len)
