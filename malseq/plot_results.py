# plot_results.py
import matplotlib.pyplot as plt
import json

# Load early detection results
results = json.load(open('early_detection_results.json'))

k_vals = [r['k'] for r in results]
base = [r['base_auroc'] for r in results]
spec = [r['spec_auroc'] for r in results]

plt.figure(figsize=(8,5))
plt.plot(k_vals, base, 'o-', label='Base (no speculation)', linewidth=2)
plt.plot(k_vals, spec, 's-', label='With speculation', linewidth=2)
plt.xlabel('Prefix Length (API calls)')
plt.ylabel('AUROC')
plt.title('Early Detection Performance')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('early_detection_plot.png', dpi=150, bbox_inches='tight')
print("✓ Saved early_detection_plot.png")
