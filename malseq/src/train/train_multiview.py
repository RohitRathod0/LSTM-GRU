import torch, os, yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ..models.multiview_model import MultiViewSeqModel
from .losses import bce_ls, next_token_loss
from ..models.calibration import TemperatureScaler
from ..models.ewc import EWCWrapper

class MVSeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def train(cfg_path, train_data, val_data, pe_dim, import_dim, device="cuda"):
    cfg = yaml.safe_load(open(cfg_path))
    model = MultiViewSeqModel(cfg, pe_dim, import_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    ds_tr = MVSeqDataset(train_data)
    ds_va = MVSeqDataset(val_data)
    tr = DataLoader(ds_tr, batch_size=cfg["training"]["batch_size"], shuffle=True)
    va = DataLoader(ds_va, batch_size=cfg["training"]["batch_size"])
    scaler = TemperatureScaler(cfg["calibration"]["temperature_init"]).to(device)

    best = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in tr:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch)
            loss_cls = bce_ls(out["logit"], batch["label"], cfg["training"]["label_smoothing"])
            loss_aux = next_token_loss(out["next_logits"], batch["next_token"])
            loss = loss_cls + cfg["heads"]["aux_nextcall_weight"] * loss_aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        # quick val
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in va:
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(device)
                out = model(batch)
                prob = torch.sigmoid(out["logit"]).cpu().numpy()
                ys.extend(batch["label"].cpu().numpy())
                ps.extend(prob)
        from .eval_metrics import evaluate_probs
        metrics = evaluate_probs(ys, ps)
        if metrics["AUROC"] > best:
            best = metrics["AUROC"]
            torch.save(model.state_dict(), "mvseq_best.pt")

    # calibration on val
    model.load_state_dict(torch.load("mvseq_best.pt", map_location=device))
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in va:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch)
            logits.append(out["logit"].detach())
            labels.append(batch["label"].float())
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    opt_cal = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)
    def closure():
        opt_cal.zero_grad()
        loss = scaler.nll_criterion(logits, labels)
        loss.backward()
        return loss
    opt_cal.step(closure)
    torch.save({"model": model.state_dict(), "temp": scaler.state_dict()}, "mvseq_calibrated.pt")
