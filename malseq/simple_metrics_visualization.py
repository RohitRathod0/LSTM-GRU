"""
simple_metrics_visualization.py - Visualize metrics from basic JSON files
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_metrics(model_name):
    """Load basic metrics"""
    path = Path(f"runs/nslkdd_{model_name.lower()}/metrics.json")
    with open(path, 'r') as f:
        return json.load(f)

def create_comparison_plots():
    """Create comparison visualizations"""
    
    models = ['LSTM', 'GRU', 'MLP']
    all_metrics = {m: load_metrics(m) for m in models}
    
    # Extract metrics
    test_acc = [all_metrics[m]['test_accuracy'] for m in models]
    val_acc = [all_metrics[m]['val_accuracy'] for m in models]
    roc_auc = [all_metrics[m]['roc_auc'] for m in models]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Accuracy Comparison
    ax1 = plt.subplot(1, 3, 1)
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, test_acc, width, label='Test Accuracy', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, val_acc, width, label='Val Accuracy', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim([0.98, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. ROC-AUC Comparison
    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.bar(models, roc_auc, color=['#e74c3c', '#9b59b6', '#f39c12'], alpha=0.8)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax2.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.99, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Overall Comparison Table
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    table_data = []
    for model in models:
        m = all_metrics[model]
        table_data.append([
            model,
            f"{m['test_accuracy']:.4f}",
            f"{m['val_accuracy']:.4f}",
            f"{m['roc_auc']:.4f}"
        ])
    
    table = ax3.table(cellText=table_data,
                     colLabels=['Model', 'Test Acc', 'Val Acc', 'ROC-AUC'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_test_idx = test_acc.index(max(test_acc))
    best_roc_idx = roc_auc.index(max(roc_auc))
    
    for i in range(len(models)):
        if i == best_test_idx:
            table[(i+1, 1)].set_facecolor('#2ecc71')
        if i == best_roc_idx:
            table[(i+1, 3)].set_facecolor('#f39c12')
    
    ax3.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = Path("runs/models_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot: {output_path}")
    
    # Print console output
    print("\n" + "="*80)
    print(" "*30 + "METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Model':<10} {'Test Accuracy':<15} {'Val Accuracy':<15} {'ROC-AUC':<10}")
    print("-" * 80)
    for model in models:
        m = all_metrics[model]
        print(f"{model:<10} {m['test_accuracy']:<15.4f} {m['val_accuracy']:<15.4f} {m['roc_auc']:<10.4f}")
    print("="*80)

if __name__ == "__main__":
    create_comparison_plots()
