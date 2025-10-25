import torch, os, yaml
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
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

def custom_collate(batch):
    """Custom collate to handle variable-length sequences"""
    pe_vecs = torch.stack([b['pe_vec'] for b in batch])
    import_vecs = torch.stack([b['import_vec'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    next_tokens = torch.stack([b['next_token'] for b in batch])
    
    tok_ids = [b['tok_ids'] for b in batch]
    arg_ids = [b['arg_ids'] for b in batch]
    bursts = [b['burst'] for b in batch]
    poss = [b['pos'] for b in batch]
    
    tok_ids_padded = rnn_utils.pad_sequence(tok_ids, batch_first=True, padding_value=0)
    arg_ids_padded = rnn_utils.pad_sequence(arg_ids, batch_first=True, padding_value=0)
    bursts_padded = rnn_utils.pad_sequence(bursts, batch_first=True, padding_value=0.0)
    poss_padded = rnn_utils.pad_sequence(poss, batch_first=True, padding_value=0.0)
    
    return {
        'pe_vec': pe_vecs,
        'import_vec': import_vecs,
        'tok_ids': tok_ids_padded,
        'arg_ids': arg_ids_padded,
        'burst': bursts_padded,
        'pos': poss_padded,
        'label': labels,
        'next_token': next_tokens
    }

def train(cfg_path, train_data, val_data, pe_dim, import_dim, device="cuda"):
    cfg = yaml.safe_load(open(cfg_path))
    model = MultiViewSeqModel(cfg, pe_dim, import_dim).to(device)
    
    lr = float(cfg["training"]["lr"])
    wd = float(cfg["training"]["weight_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    ds_tr = MVSeqDataset(train_data)
    ds_va = MVSeqDataset(val_data)
    tr = DataLoader(ds_tr, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=custom_collate)
    va = DataLoader(ds_va, batch_size=cfg["training"]["batch_size"], collate_fn=custom_collate)
    
    scaler = TemperatureScaler(cfg["calibration"]["temperature_init"]).to(device)

    best = 0.0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']}: Loss={epoch_loss/len(tr):.4f}")

        # Quick val
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
        print(f"  Val AUROC: {metrics['AUROC']:.4f}")
        
        if metrics["AUROC"] > best:
            best = metrics["AUROC"]
            torch.save(model.state_dict(), "mvseq_best.pt")
            print(f"  ✓ New best model saved!")

    # Calibration on val
    print("\nCalibrating temperature...")
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
    print("✓ Calibration complete! Saved mvseq_calibrated.pt")
