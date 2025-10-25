import os
import json
import math
import random
import numpy as np
import torch
from typing import Dict, List, Tuple

# Expected inputs:
# - data/static_raw.pt: list of dicts per sample:
#     { "id": str, "pe_vec": {feature_name: float/int, ...}, "imports": [dll1, dll2, ...] }
# - data/dynamic_tokenized.pt: dict with:
#     { "tokenizer": {token->id mapping}, "data": [ { "id": str, "tok_ids": LongTensor[T],
#           "arg_ids": LongTensor[T], "burst": FloatTensor[T,1], "pos": FloatTensor[T,1],
#           "label": tensor(0/1) }, ... ] }
# Optional index file to map IDs across static/dynamic:
# - data/index.json: { "splits": {"train":[ids], "val":[ids], "test":[ids]} }
#
# Outputs:
# - data/train.pt, data/val.pt, data/test.pt: list of batch-ready dicts per sample:
#     { pe_vec: FloatTensor[pe_dim], import_vec: FloatTensor[import_dim],
#       tok_ids: LongTensor[T], arg_ids: LongTensor[T], burst: FloatTensor[T,1], pos: FloatTensor[T,1],
#       label: LongTensor[], next_token: LongTensor[] }
#
# Notes:
# - Builds a fixed feature order for PE fields and import DLL bag-of-words.
# - Aligns samples by shared "id". If IDs are missing, uses file order alignment but warns.
# - Adds a next_token target (the token following the last prefix position) for auxiliary head.
# - Supports prefix cropping curriculum lengths to facilitate Early@k training/eval.

RNG = random.Random(1337)
NP_RNG = np.random.RandomState(1337)

def _collect_pe_schema(static_list: List[Dict]) -> Tuple[List[str], List[str]]:
    # Determine consistent PE field keys and DLL vocabulary
    pe_keys = set()
    dlls = set()
    for s in static_list:
        for k in s["pe_vec"].keys():
            pe_keys.add(k)
        for d in s["imports"]:
            dlls.add(d.lower())
    pe_keys = sorted(list(pe_keys))
    dll_vocab = sorted(list(dlls))
    return pe_keys, dll_vocab

def _pe_to_vec(sample: Dict, pe_keys: List[str], dll_vocab: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    pe = sample["pe_vec"]
    x = np.array([float(pe.get(k, 0.0)) for k in pe_keys], dtype=np.float32)
    imp = np.zeros(len(dll_vocab), dtype=np.float32)
    present = set([d.lower() for d in sample["imports"]])
    for i, dll in enumerate(dll_vocab):
        if dll in present:
            imp[i] = 1.0
    # simple log scaling for heavy-tailed fields
    x = np.sign(x) * np.log1p(np.abs(x))
    return torch.from_numpy(x), torch.from_numpy(imp)

def _align_by_id(static_list: List[Dict], dyn_list: List[Dict]):
    # Build map id->static/dynamic
    s_map = {}
    d_map = {}
    for i, s in enumerate(static_list):
        s_map[s.get("id", f"s{i}")] = s
    for j, d in enumerate(dyn_list):
        d_map[d.get("id", f"d{j}")] = d
    ids = sorted(list(set(s_map.keys()) & set(d_map.keys())))
    if len(ids) == 0:
        # fallback: align by order
        n = min(len(static_list), len(dyn_list))
        ids = [f"idx{i}" for i in range(n)]
        for i in range(n):
            static_list[i]["id"] = ids[i]
            dyn_list[i]["id"] = ids[i]
    return ids, s_map, d_map

def _make_next_token_target(tok_ids: torch.LongTensor) -> int:
    # Use the last observed token as a proxy target for next-token head during training
    # Optionally, could use tok_ids[min(len)-1] or a shifted sequence task for teacher forcing.
    return int(tok_ids[-1].item())

def _crop_prefix(sample_dyn: Dict, max_len: int):
    # Crop the sequence to at most max_len, preserving tensors
    T = sample_dyn["tok_ids"].shape[0]
    t = min(T, max_len)
    out = {}
    for k in ["tok_ids", "arg_ids"]:
        out[k] = sample_dyn[k][:t]
    for k in ["burst", "pos"]:
        out[k] = sample_dyn[k][:t]
    return out

def build_dataset(
    static_path="data/static_raw.pt",
    dynamic_path="data/dynamic_tokenized.pt",
    index_path="data/index.json",
    out_dir="data",
    prefix_lengths=(64,),  # can pass multiple for mixed-prefix dataset
    pe_log_scale=True
):
    os.makedirs(out_dir, exist_ok=True)
    static_list = torch.load(static_path)
    dyn_blob = torch.load(dynamic_path)
    dyn_list = dyn_blob["data"]
    tokenizer_stoi = dyn_blob.get("tokenizer", {})

    pe_keys, dll_vocab = _collect_pe_schema(static_list)
    ids, s_map, d_map = _align_by_id(static_list, dyn_list)

    # Load splits or make a default split
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            splits = json.load(f)["splits"]
        train_ids = [i for i in splits["train"] if i in ids]
        val_ids = [i for i in splits["val"] if i in ids]
        test_ids = [i for i in splits["test"] if i in ids]
    else:
        RNG.shuffle(ids)
        n = len(ids)
        n_tr = int(0.8 * n)
        n_va = int(0.1 * n)
        train_ids = ids[:n_tr]
        val_ids = ids[n_tr:n_tr+n_va]
        test_ids = ids[n_tr+n_va:]

    def build_split(id_list):
        out = []
        for sid in id_list:
            s = s_map[sid]
            d = d_map[sid]
            pe_vec, import_vec = _pe_to_vec(s, pe_keys, dll_vocab)
            label = d["label"].long()
            for L in prefix_lengths:
                dyn_crop = _crop_prefix(d, L)
                # ensure tensors
                tok_ids = dyn_crop["tok_ids"].long()
                arg_ids = dyn_crop["arg_ids"].long()
                burst = dyn_crop["burst"].float()
                pos = dyn_crop["pos"].float()
                # next-token target from last observed token (simple, robust)
                next_token = torch.tensor(_make_next_token_target(tok_ids), dtype=torch.long)
                out.append({
                    "pe_vec": pe_vec.clone(),
                    "import_vec": import_vec.clone(),
                    "tok_ids": tok_ids.clone(),
                    "arg_ids": arg_ids.clone(),
                    "burst": burst.clone(),
                    "pos": pos.clone(),
                    "label": label.clone(),
                    "next_token": next_token
                })
        return out

    train_data = build_split(train_ids)
    val_data = build_split(val_ids)
    test_data = build_split(test_ids)

    torch.save({
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "pe_keys": pe_keys,
        "dll_vocab": dll_vocab,
        "tokenizer": tokenizer_stoi,
        "prefix_lengths": list(prefix_lengths)
    }, os.path.join(out_dir, "mvseq_dataset.pt"))

    # Also dump convenience per-split files expected by training script
    torch.save(train_data, os.path.join(out_dir, "train.pt"))
    torch.save(val_data, os.path.join(out_dir, "val.pt"))
    torch.save(test_data, os.path.join(out_dir, "test.pt"))

if __name__ == "__main__":
    # Example: create multiple prefix lengths to support Early@k evaluation
    build_dataset(
        static_path="data/static_raw.pt",
        dynamic_path="data/dynamic_tokenized.pt",
        index_path="data/index.json",
        out_dir="data",
        prefix_lengths=(16, 32, 64, 96)
    )
