import os, json
import torch
import numpy as np

# 1) Point to your dataset once
CSV = r"C:\Users\rohit\OneDrive\Desktop\LSTM\data\archive\Unicauca-dataset-April-June-2019-Network-flows.csv"
LABEL = "ApplicationName"   # TODO: change after checking columns if different
OUTDIR = "runs/unicauca"
EPOCHS = 8
BATCH = 2048
LR = 3e-4
WD = 1e-4
MAX_ROWS = 150000   # set None to train full

# 2) Call the existing training function programmatically
import train_flows as TF

def main():
    # If your train_flows.py doesn’t expose a function, it already does (train).
    # We need a small modification in train_flows.py to allow passing label:
    # - Add `label_col` param to load_data(...) call.
    # Edit train_flows.py:
    #   def load_data(csv_path, label_col="application", ...)
    #   ...
    #   Xtr,... = load_data(csv_path, label_col=label_col, ...)
    #
    # If you’ve added argparse only, we can temporarily monkeypatch the default by setting TF.load_data via wrapper:
    def load_with_label(csv_path, **kw):
        kw.setdefault("label_col", LABEL)
        return TF.load_data(csv_path, **kw)
    TF.load_data = load_with_label  # ensure our label is used

    os.makedirs(OUTDIR, exist_ok=True)
    TF.train(CSV, outdir=OUTDIR, epochs=EPOCHS, batch=BATCH, lr=LR, wd=WD, max_rows=MAX_ROWS)

    # 3) Print checkpoint summary
    ck = torch.load(os.path.join(OUTDIR, "calibrated.pt"), map_location="cpu")
    print(f"\n✓ Saved to {OUTDIR}")
    print("Classes:", len(ck["label_map"]))
    T = float(torch.exp(ck["temp"]["logT"]))
    print("Temperature:", T)

    # Optional: show metrics.json
    mpath = os.path.join(OUTDIR, "metrics.json")
    if os.path.exists(mpath):
        print("\nMetrics:", json.dumps(json.load(open(mpath)), indent=2))

if __name__ == "__main__":
    main()
