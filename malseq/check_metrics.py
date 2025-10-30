import json
from pathlib import Path

# Check what's in the metrics file
metrics_path = Path("runs/nslkdd_lstm/metrics.json")
with open(metrics_path, 'r') as f:
    data = json.load(f)

print("Keys in metrics.json:")
for key in data.keys():
    print(f"  - {key}")

print("\nFull structure:")
print(json.dumps(data, indent=2))
