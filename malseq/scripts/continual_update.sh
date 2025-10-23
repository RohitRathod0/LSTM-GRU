#!/usr/bin/env bash
python - <<'PY'
import torch, yaml
from torch.utils.data import DataLoader, Dataset
from src.models.multiview_model import MultiViewSeqModel
from src.models.ewc import EWCWrapper
from src.train.losses import bce_ls
cfg = yaml.safe_load(open("configs/mv_gru_default.yaml"))
device = "cuda"
model = MultiViewSeqModel(cfg, pe_dim=64, import_dim=512).to(device)
ckpt = torch.load("mvseq_calibrated.pt", map_location=device)
model.load_state_dict(ckpt["model"])
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["continual"]["lr"])
# Freeze most layers; update gate + last GRU + classifier
for n, p in model.named_parameters():
    p.requires_grad = any(k in n for k in ["gate", "seq_gru", "cls_"])
# Build small labeled recent dataset loader
recent = torch.load("data/recent_buffer.pt")  # list of batches
class D(Dataset):
    def __len__(self): return len(recent)
    def __getitem__(self, i): return recent[i]
dl = DataLoader(D(), batch_size=64, shuffle=True)
ewc = EWCWrapper(model, lambda_=cfg["continual"]["ewc_lambda"])
# Estimate Fisher on prior val subset
prior = torch.load("data/val_small.pt")
class V(Dataset):
    def __len__(self): return len(prior)
    def __getitem__(self, i): return prior[i]
ewc.compute_fisher(DataLoader(V(), batch_size=64), lambda logits, y: bce_ls(logits, y, 0.0), device)

model.train()
for epoch in range(2):
    for batch in dl:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        out = model(batch)
        loss = bce_ls(out["logit"], batch["label"], 0.0) + ewc.penalty()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
torch.save(model.state_dict(), "mvseq_continual.pt")
PY
