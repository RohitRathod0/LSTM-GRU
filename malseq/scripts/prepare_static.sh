#!/usr/bin/env bash
# Example: batch PE feature extraction to parquet/pt
python - <<'PY'
import os, torch, json
from src.data.extract_pe import extract_pe_features
import numpy as np
root = "samples/pe/"
out = []
for f in os.listdir(root):
    if not f.lower().endswith(".exe"): continue
    feats, imports = extract_pe_features(os.path.join(root, f))
    out.append({"pe_vec": feats, "imports": imports})
torch.save(out, "data/static_raw.pt")
PY
