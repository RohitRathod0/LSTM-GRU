import sys
sys.path.insert(0, '.')

import torch
import yaml
from src.models.multiview_model import MultiViewSeqModel
from src.models.calibration import TemperatureScaler
from src.train.eval_metrics import evaluate_probs

print("="*60)
print("EVALUATION: Multi-View Malware Detector")
print("="*60)

# Load config and detect dimensions
cfg = yaml.safe_load(open('configs/mv_gru_default.yaml'))
sample = torch.load('data/test.pt')[0]
pe_dim = sample['pe_vec'].shape[0]
import_dim = sample['import_vec'].shape[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
print(f"PE feature dim: {pe_dim}, Import feature dim: {import_dim}")

# Load calibrated model
print("\nLoading calibrated model from mvseq_calibrated.pt...")
model = MultiViewSeqModel(cfg, pe_dim, import_dim).to(device)
ckpt = torch.load('mvseq_calibrated.pt', map_location=device)
model.load_state_dict(ckpt['model'])
temp = TemperatureScaler()
temp.load_state_dict(ckpt['temp'])
model.eval()
print("✓ Model loaded successfully")

# Evaluate on test set
print("\nEvaluating on test set...")
test_data = torch.load('data/test.pt')
print(f"Test samples: {len(test_data)}")

ys, ps, gates = [], [], []

with torch.no_grad():
    for i, sample in enumerate(test_data):
        if (i+1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_data)} samples...", end='\r')
        
        # Convert to batch format
        batch = {
            'pe_vec': sample['pe_vec'].unsqueeze(0).to(device),
            'import_vec': sample['import_vec'].unsqueeze(0).to(device),
            'tok_ids': sample['tok_ids'].unsqueeze(0).to(device),
            'arg_ids': sample['arg_ids'].unsqueeze(0).to(device),
            'burst': sample['burst'].unsqueeze(0).to(device),
            'pos': sample['pos'].unsqueeze(0).to(device),
            'label': sample['label'],
            'next_token': sample['next_token']
        }
        
        out = model(batch)
        scaled = temp(out['logit'])
        prob = torch.sigmoid(scaled).item()
        gate_val = out['gate'].item()
        
        ys.append(int(sample['label'].item()))
        ps.append(prob)
        gates.append(gate_val)

print(f"  Processed {len(test_data)}/{len(test_data)} samples     ")

# Compute metrics
results = evaluate_probs(ys, ps)

print("\n" + "="*60)
print("TEST SET RESULTS")
print("="*60)
for k, v in results.items():
    print(f"  {k:20s}: {v:.4f}")

print(f"\n  Total samples:        {len(ys)}")
print(f"  Malware samples:      {sum(ys)} ({100*sum(ys)/len(ys):.1f}%)")
print(f"  Benign samples:       {len(ys)-sum(ys)} ({100*(len(ys)-sum(ys))/len(ys):.1f}%)")

# Gate statistics (fusion analysis)
import numpy as np
gates = np.array(gates)
print(f"\n  Fusion gate stats:")
print(f"    Mean gate value:    {gates.mean():.3f}")
print(f"    Std gate value:     {gates.std():.3f}")
print(f"    (gate=0: static, gate=1: dynamic)")

print("\n" + "="*60)
print("✓ Evaluation complete!")
print("="*60)

# Save results
import json
with open('eval_results.json', 'w') as f:
    json.dump({
        'metrics': results,
        'gate_mean': float(gates.mean()),
        'gate_std': float(gates.std()),
        'n_samples': len(ys),
        'n_malware': sum(ys)
    }, f, indent=2)
print("\n✓ Results saved to eval_results.json")
