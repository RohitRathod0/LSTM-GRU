"""
detailed_metrics_analysis.py - Works with basic metrics.json and generates visualizations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

def load_metrics(model_name, base_dir="runs"):
    """Load metrics from JSON file"""
    metrics_path = Path(base_dir) / f"nslkdd_{model_name.lower()}" / "metrics.json"
    with open(metrics_path, 'r') as f:
        return json.load(f)

def create_all_visualizations(all_metrics, output_dir="runs"):
    """Create comprehensive comparison visualizations"""
    
    models = list(all_metrics.keys())
    
    # Extract metrics
    test_acc = [all_metrics[m]['test_accuracy'] for m in models]
    val_acc = [all_metrics[m]['val_accuracy'] for m in models]
    roc_auc = [all_metrics[m]['roc_auc'] for m in models]
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Test Accuracy Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(models, test_acc, color=['#e74c3c', '#9b59b6', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(test_acc) - 0.005, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Validation Accuracy Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(models, val_acc, color=['#3498db', '#2ecc71', '#e67e22'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([min(val_acc) - 0.005, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. ROC-AUC Comparison
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(models, roc_auc, color=['#1abc9c', '#34495e', '#c0392b'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
    ax3.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim([min(roc_auc) - 0.001, 1.0])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Test vs Validation Accuracy
    ax4 = plt.subplot(2, 3, 4)
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, test_acc, width, label='Test', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, val_acc, width, label='Validation', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Test vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend(fontsize=11)
    ax4.set_ylim([min(min(test_acc), min(val_acc)) - 0.005, 1.0])
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Overfitting Analysis (Test - Val gap)
    ax5 = plt.subplot(2, 3, 5)
    gaps = [test_acc[i] - val_acc[i] for i in range(len(models))]
    colors_gap = ['#2ecc71' if g > 0 else '#e74c3c' for g in gaps]
    bars = ax5.bar(models, gaps, color=colors_gap, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_ylabel('Accuracy Gap', fontsize=12, fontweight='bold')
    ax5.set_title('Generalization Gap (Test - Val)', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.4f}\n({abs(gap)*100:.2f}%)',
                ha='center', va='bottom' if gap > 0 else 'top', fontsize=9, fontweight='bold')
    
    # 6. Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = []
    for model in models:
        m = all_metrics[model]
        table_data.append([
            model,
            f"{m['test_accuracy']:.4f}",
            f"{m['val_accuracy']:.4f}",
            f"{m['roc_auc']:.4f}",
            f"{m['test_accuracy'] - m['val_accuracy']:.4f}"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'Test Acc', 'Val Acc', 'ROC-AUC', 'Gap'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_test_idx = test_acc.index(max(test_acc))
    best_val_idx = val_acc.index(max(val_acc))
    best_roc_idx = roc_auc.index(max(roc_auc))
    
    for i in range(len(models)):
        if i == best_test_idx:
            table[(i+1, 1)].set_facecolor('#2ecc71')
            table[(i+1, 1)].set_text_props(weight='bold')
        if i == best_val_idx:
            table[(i+1, 2)].set_facecolor('#3498db')
            table[(i+1, 2)].set_text_props(weight='bold', color='white')
        if i == best_roc_idx:
            table[(i+1, 3)].set_facecolor('#f39c12')
            table[(i+1, 3)].set_text_props(weight='bold')
    
    ax6.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "comprehensive_metrics_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comprehensive dashboard: {output_path}")
    
    # Create individual metric plots
    create_individual_plots(all_metrics, output_dir)

def create_individual_plots(all_metrics, output_dir):
    """Create separate detailed plots for each metric"""
    
    models = list(all_metrics.keys())
    test_acc = [all_metrics[m]['test_accuracy'] for m in models]
    val_acc = [all_metrics[m]['val_accuracy'] for m in models]
    roc_auc = [all_metrics[m]['roc_auc'] for m in models]
    
    # 1. Accuracy Comparison Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, test_acc, width, label='Test Accuracy', 
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, val_acc, width, label='Validation Accuracy',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([min(min(test_acc), min(val_acc)) - 0.01, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved accuracy comparison: {output_path}")
    
    # 2. ROC-AUC Detailed Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#e74c3c', '#9b59b6', '#f39c12']
    bars = ax.bar(models, roc_auc, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('ROC-AUC Score', fontsize=14, fontweight='bold')
    ax.set_title('ROC-AUC Score Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([min(roc_auc) - 0.002, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
               f'{height:.6f}', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "roc_auc_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC-AUC comparison: {output_path}")
    
    # 3. Combined Line Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(models, test_acc, 'o-', linewidth=3, markersize=12, 
            label='Test Accuracy', color='#2ecc71', markeredgecolor='black', markeredgewidth=2)
    ax.plot(models, val_acc, 's-', linewidth=3, markersize=12,
            label='Val Accuracy', color='#3498db', markeredgecolor='black', markeredgewidth=2)
    ax.plot(models, roc_auc, '^-', linewidth=3, markersize=12,
            label='ROC-AUC', color='#e74c3c', markeredgecolor='black', markeredgewidth=2)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('All Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0.985, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "all_metrics_line_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved combined line plot: {output_path}")

def main():
    print("\n" + "="*80)
    print(" "*25 + "DETAILED METRICS ANALYSIS")
    print("="*80)
    
    models = ['LSTM', 'GRU', 'MLP']
    all_metrics = {}
    
    # Load metrics for each model
    for model in models:
        print(f"\n{'='*80}")
        print(f"{model} - Model Evaluation")
        print(f"{'='*80}")
        
        metrics = load_metrics(model)
        all_metrics[model] = metrics
        
        print(f"\n✅ Metrics:")
        print(f"   Test Accuracy:       {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        print(f"   Validation Accuracy: {metrics['val_accuracy']:.4f} ({metrics['val_accuracy']*100:.2f}%)")
        print(f"   ROC-AUC Score:       {metrics['roc_auc']:.4f}")
        print(f"   Generalization Gap:  {metrics['test_accuracy'] - metrics['val_accuracy']:.4f}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(" "*30 + "COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<10} │ {'Test Acc':<12} │ {'Val Acc':<12} │ {'ROC-AUC':<12} │ {'Gap':<10}")
    print("-" * 80)
    
    for model in models:
        m = all_metrics[model]
        gap = m['test_accuracy'] - m['val_accuracy']
        print(f"{model:<10} │ {m['test_accuracy']:<12.4f} │ {m['val_accuracy']:<12.4f} │ "
              f"{m['roc_auc']:<12.4f} │ {gap:<10.4f}")
    
    print("="*80)
    
    # Identify best model
    best_test = max(all_metrics.items(), key=lambda x: x[1]['test_accuracy'])
    best_roc = max(all_metrics.items(), key=lambda x: x[1]['roc_auc'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   Highest Test Accuracy: {best_test[0]} ({best_test[1]['test_accuracy']:.4f})")
    print(f"   Highest ROC-AUC:       {best_roc[0]} ({best_roc[1]['roc_auc']:.6f})")
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")
    
    create_all_visualizations(all_metrics)
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}")
    
    print("\n📁 Generated visualization files:")
    print("   • comprehensive_metrics_dashboard.png - 6-panel overview")
    print("   • accuracy_comparison.png - Test vs Val accuracy")
    print("   • roc_auc_comparison.png - ROC-AUC scores")
    print("   • all_metrics_line_plot.png - Combined line chart\n")

if __name__ == "__main__":
    main()
