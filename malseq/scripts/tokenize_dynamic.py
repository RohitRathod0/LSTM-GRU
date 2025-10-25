import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
print(f"Added to path: {project_root}")

import json
import torch
from src.data.tokenizers import APITokenizer, ArgBinner

# Rest of your code
raw = json.load(open("data/dynamic_raw.json"))
tok = APITokenizer(max_vocab=4096)
tok.fit([x["apis"] for x in raw])
argb = ArgBinner()

processed = []
for x in raw:
    ti = tok.encode(x["apis"])
    ai = argb.encode_args(x["args"])
    tok_ids = torch.tensor(ti, dtype=torch.long)
    arg_ids = torch.tensor([a[0] for a in ai], dtype=torch.long)
    T = len(ti)
    burst = torch.ones(T, 1)
    pos = torch.arange(T).float().unsqueeze(-1)
    processed.append({
        "id": x["id"], 
        "tok_ids": tok_ids, 
        "arg_ids": arg_ids,
        "burst": burst, 
        "pos": pos, 
        "label": torch.tensor(x["label"])
    })

torch.save({"tokenizer": tok.stoi, "data": processed}, "data/dynamic_tokenized.pt")
print(f"Tokenized {len(processed)} sequences; vocab={len(tok.stoi)}")
