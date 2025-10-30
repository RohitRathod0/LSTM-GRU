import os, glob, torch
from src.data.extract_pe import extract_pe_features

def scan_dir(root):
    out = []
    for fp in glob.glob(os.path.join(root, "**/*.*"), recursive=True):
        if fp.lower().endswith((".exe",".dll",".sys",".ocx")):
            try:
                feats, imports = extract_pe_features(fp)
                out.append({"id": os.path.splitext(os.path.basename(fp))[0],
                            "pe_vec": feats, "imports": imports})
            except Exception:
                continue
    return out

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    # Replace with your benign/malware folders
    pe_samples = scan_dir("data/raw/pe_samples")
    torch.save(pe_samples, "data/static_raw.pt")
    print(f"Extracted static features for {len(pe_samples)} PEs")
