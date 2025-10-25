import os, json, re, torch
from pathlib import Path

def load_ocatak(root):
    # Expect: malware_api_class/* contains per-family folders with sequences in text
    samples = []
    for label_dir, y in [("Malware", 1), ("Benign", 0)]:
        p = Path(root)/label_dir
        if not p.exists(): continue
        for fp in p.rglob("*.txt"):
            try:
                lines = [l.strip() for l in open(fp, errors="ignore").read().splitlines() if l.strip()]
                apis = []
                args = []
                for ln in lines:
                    # Example format "CreateFileW(size=1024, flags=3)" or "CreateFileW"
                    m = re.match(r"([A-Za-z0-9_]+)\((.*)\)", ln)
                    if m:
                        api = m.group(1)
                        argstr = m.group(2)
                        size = 0
                        flags = 0
                        ms = re.search(r"size\s*=\s*([0-9]+)", argstr)
                        if ms: size = int(ms.group(1))
                        mf = re.search(r"flags\s*=\s*([0-9]+)", argstr)
                        if mf: flags = int(mf.group(1))
                    else:
                        api = ln
                        size = 0
                        flags = 0
                    apis.append(api)
                    args.append({"size": size, "flags": flags})
                if len(apis) >= 8:
                    samples.append({"id": fp.stem, "apis": apis, "args": args, "label": y})
            except Exception:
                continue
    return samples

if __name__ == "__main__":
    data = load_ocatak("data/raw/malware_api_class")
    os.makedirs("data", exist_ok=True)
    json.dump(data, open("data/dynamic_raw.json","w"))
    print(f"Parsed {len(data)} dynamic samples")
