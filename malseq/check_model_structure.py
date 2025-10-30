import torch

checkpoint = torch.load('runs/nslkdd_lstm/best.pt', map_location='cpu')

# Check what's in the checkpoint
print("Checkpoint keys:", checkpoint.keys())
print("\nModel structure:")
for key, value in checkpoint.items():
    if 'state_dict' in key.lower():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v.shape}")
    elif isinstance(value, dict) and any('weight' in k or 'bias' in k for k in value.keys()):
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v.shape}")
