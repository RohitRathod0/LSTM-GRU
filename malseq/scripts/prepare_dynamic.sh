#!/usr/bin/env bash
# Example: tokenize API call sequences from sandbox logs
python - <<'PY'
import torch, json, os
from src.data.tokenizers import APITokenizer, ArgBinner

logs = []  # fill with parsed sandbox sequences [{'apis':[...], 'args':[...], 'label':0/1}, ...]
# e.g., ingest datasets similar in structure to Kaggle API call sequences
# (ensure you own the data rights/ethics)
tok = APITokenizer(max_vocab=4096)
tok.fit([x["apis"] for x in logs])
argb = ArgBinner()

data = []
for x in logs:
    ti = tok.encode(x["apis"])
    ai = [(a[0], a[1]) for a in argb.encode_args(x["args"])]
    # pack tensors
    import torch
    tok_ids = torch.tensor(ti, dtype=torch.long)
    arg_ids = torch.tensor([a[0] for a in ai], dtype=torch.long)
    burst = torch.ones(len(ti), 1)
    pos = torch.arange(len(ti)).float().unsqueeze(1)
    data.append({"tok_ids": tok_ids, "arg_ids": arg_ids, "burst": burst, "pos": pos, "label": torch.tensor(x["label"])})
torch.save({"tokenizer": tok.stoi, "data": data}, "data/dynamic_tokenized.pt")
PY
