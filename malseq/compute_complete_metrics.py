"""
compute_complete_metrics.py - Load models and compute comprehensive metrics (FIXED)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# CORRECTED Model architectures matching your trained models
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_test_data():
    """Load preprocessed test data"""
    test_path = Path('data/nslkdd_test_preprocessed.npz')
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    data = np.load(test_path)
    X_test = torch.FloatTensor(data['X'])
    y_test = torch.LongTensor(data['y'])
    
    print(f"✓ Loaded test data: {X_test.shape}")
    return X_test, y_test

def evaluate_model(model_name, model, X_test, y_test, device='cpu'):
    """Evaluate model and compute comprehensive metrics"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name.upper()}...")
    print(f"{'='*80}")
    
    model.eval()
    model.to(device)
    
    # Get predictions
    with torch.no_grad():
        if model_name in ['LSTM', 'GRU']:
            # Reshape for sequence models
            if len(X_test.shape) == 2:
                X_test = X_test.unsqueeze(1)
        
        X_test = X_test.to(device)
        outputs = model(X_test)
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        y_proba = probs[:, 1].cpu().numpy()
        
        # Get predictions
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    
    y_true = y_test.cpu().numpy()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Compute overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)
    
    # Compute ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Compute rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Build comprehensive metrics
    metrics = {
        "model": model_name.lower(),
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "Normal": {
                "label": 0,
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0])
            },
            "Malware": {
                "label": 1,
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1])
            }
        },
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "tpr": float(tpr)
    }
    
    # Save metrics
    output_path = Path(f"runs/nslkdd_{model_name.lower()}/complete_metrics.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved complete metrics to: {output_path}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision (macro):  {precision_macro:.4f}")
    print(f"   Recall (macro):     {recall_macro:.4f}")
    print(f"   F1-Score (macro):   {f1_macro:.4f}")
    print(f"   ROC-AUC:            {roc_auc:.4f}")
    print(f"   PR-AUC:             {pr_auc:.4f}")
    print(f"\n   Malware Detection:")
    print(f"   ├─ Precision: {precision[1]:.4f}")
    print(f"   ├─ Recall:    {recall[1]:.4f}")
    print(f"   ├─ F1-Score:  {f1[1]:.4f}")
    print(f"   └─ FNR:       {fnr:.4f} ({fnr*100:.2f}%)")
    
    return metrics

def create_comprehensive_visualizations(all_metrics, output_dir="runs"):
    """Generate comprehensive visualization dashboard"""
    
    models = list(all_metrics.keys())
    
    # Extract all metrics
    test_acc = [all_metrics[m]['test_accuracy'] for m in models]
    precision_macro = [all_metrics[m]['precision_macro'] for m in models]
    recall_macro = [all_metrics[m]['recall_macro'] for m in models]
    f1_macro = [all_metrics[m]['f1_macro'] for m in models]
    roc_auc = [all_metrics[m]['roc_auc'] for m in models]
    
    # Malware-specific
    precision_malware = [all_metrics[m]['per_class']['Malware']['precision'] for m in models]
    recall_malware = [all_metrics[m]['per_class']['Malware']['recall'] for m in models]
    f1_malware = [all_metrics[m]['per_class']['Malware']['f1'] for m in models]
    
    # Error rates
    fpr = [all_metrics[m]['fpr'] * 100 for m in models]
    fnr = [all_metrics[m]['fnr'] * 100 for m in models]
    
    # Create mega dashboard
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overall Metrics
    ax1 = plt.subplot(2, 4, 1)
    x = np.arange(len(models))
    width = 0.18
    ax1.bar(x - 2*width, test_acc, width, label='Accuracy', color='#2ecc71', edgecolor='black')
    ax1.bar(x - width, precision_macro, width, label='Precision', color='#3498db', edgecolor='black')
    ax1.bar(x, recall_macro, width, label='Recall', color='#e74c3c', edgecolor='black')
    ax1.bar(x + width, f1_macro, width, label='F1-Score', color='#f39c12', edgecolor='black')
    ax1.bar(x + 2*width, roc_auc, width, label='ROC-AUC', color='#9b59b6', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax1.set_title('Overall Metrics Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim([0.93, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Malware Detection Metrics
    ax2 = plt.subplot(2, 4, 2)
    width = 0.25
    ax2.bar(x - width, precision_malware, width, label='Precision', color='#3498db', edgecolor='black')
    ax2.bar(x, recall_malware, width, label='Recall', color='#e74c3c', edgecolor='black')
    ax2.bar(x + width, f1_malware, width, label='F1-Score', color='#f39c12', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax2.set_title('Malware Detection Metrics', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0.93, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Error Rates
    ax3 = plt.subplot(2, 4, 3)
    width = 0.35
    ax3.bar(x - width/2, fpr, width, label='False Positive Rate', color='#e74c3c', edgecolor='black')
    ax3.bar(x + width/2, fnr, width, label='False Negative Rate', color='#c0392b', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.set_ylabel('Error Rate (%)', fontweight='bold', fontsize=11)
    ax3.set_title('Error Rates (Lower = Better)', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. F1-Score by Class
    ax4 = plt.subplot(2, 4, 4)
    f1_normal = [all_metrics[m]['per_class']['Normal']['f1'] for m in models]
    width = 0.35
    ax4.bar(x - width/2, f1_normal, width, label='Normal', color='#2ecc71', edgecolor='black')
    ax4.bar(x + width/2, f1_malware, width, label='Malware', color='#e74c3c', edgecolor='black')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.set_ylabel('F1-Score', fontweight='bold', fontsize=11)
    ax4.set_title('F1-Score by Class', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_ylim([0.93, 1.0])
    ax4.grid(axis='y', alpha=0.3)
    
    # 5-7. Confusion Matrices
    for idx, model in enumerate(models):
        ax = plt.subplot(2, 4, 5 + idx)
        cm = np.array(all_metrics[model]['confusion_matrix'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Malware'],
                    yticklabels=['Normal', 'Malware'],
                    cbar_kws={'label': 'Rate'},
                    ax=ax, square=True, cbar=False)
        ax.set_title(f'{model} Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
    
    # 8. Summary Table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    table_data = []
    for model in models:
        m = all_metrics[model]
        table_data.append([
            model,
            f"{m['test_accuracy']:.4f}",
            f"{m['f1_macro']:.4f}",
            f"{m['per_class']['Malware']['recall']:.4f}",
            f"{m['fnr']*100:.2f}%"
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Model', 'Accuracy', 'F1', 'Recall', 'FNR'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax8.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "complete_metrics_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved comprehensive dashboard: {output_path}")
    
    # Create individual plots
    create_individual_plots(all_metrics, output_dir)

def create_individual_plots(all_metrics, output_dir):
    """Create separate detailed plots for each metric type"""
    
    models = list(all_metrics.keys())
    x = np.arange(len(models))
    width = 0.35
    
    # 1. F1-Score Detailed
    fig, ax = plt.subplots(figsize=(12, 7))
    f1_normal = [all_metrics[m]['per_class']['Normal']['f1'] for m in models]
    f1_malware = [all_metrics[m]['per_class']['Malware']['f1'] for m in models]
    
    bars1 = ax.bar(x - width/2, f1_normal, width, label='Normal', color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, f1_malware, width, label='Malware', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
    ax.set_title('F1-Score Comparison by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0.9, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "f1_score_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved F1-Score comparison")
    
    # 2. Precision Detailed
    fig, ax = plt.subplots(figsize=(12, 7))
    prec_normal = [all_metrics[m]['per_class']['Normal']['precision'] for m in models]
    prec_malware = [all_metrics[m]['per_class']['Malware']['precision'] for m in models]
    
    bars1 = ax.bar(x - width/2, prec_normal, width, label='Normal', color='#1abc9c', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, prec_malware, width, label='Malware', color='#e67e22', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision Comparison by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0.9, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "precision_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Precision comparison")
    
    # 3. Recall Detailed
    fig, ax = plt.subplots(figsize=(12, 7))
    rec_normal = [all_metrics[m]['per_class']['Normal']['recall'] for m in models]
    rec_malware = [all_metrics[m]['per_class']['Malware']['recall'] for m in models]
    
    bars1 = ax.bar(x - width/2, rec_normal, width, label='Normal', color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, rec_malware, width, label='Malware', color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=14, fontweight='bold')
    ax.set_title('Recall Comparison by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0.9, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "recall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Recall comparison")

def main():
    print("\n" + "="*80)
    print(" "*20 + "COMPLETE METRICS COMPUTATION & VISUALIZATION")
    print("="*80)
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Get input size
    input_size = X_test.shape[1]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    all_metrics = {}
    
    # LSTM
    try:
        print("Loading LSTM model...")
        lstm_model = LSTMModel(input_size=input_size)
        checkpoint = torch.load('runs/nslkdd_lstm/best.pt', map_location=device)
        lstm_model.load_state_dict(checkpoint)
        metrics_lstm = evaluate_model('LSTM', lstm_model, X_test, y_test, device)
        all_metrics['LSTM'] = metrics_lstm
    except Exception as e:
        print(f"❌ Error loading LSTM: {e}")
    
    # GRU
    try:
        print("\nLoading GRU model...")
        gru_model = GRUModel(input_size=input_size)
        checkpoint = torch.load('runs/nslkdd_gru/best.pt', map_location=device)
        gru_model.load_state_dict(checkpoint)
        metrics_gru = evaluate_model('GRU', gru_model, X_test, y_test, device)
        all_metrics['GRU'] = metrics_gru
    except Exception as e:
        print(f"❌ Error loading GRU: {e}")
    
    # MLP
    try:
        print("\nLoading MLP model...")
        mlp_model = MLPModel(input_size=input_size)
        checkpoint = torch.load('runs/nslkdd_mlp/best.pt', map_location=device)
        mlp_model.load_state_dict(checkpoint)
        metrics_mlp = evaluate_model('MLP', mlp_model, X_test, y_test, device)
        all_metrics['MLP'] = metrics_mlp
    except Exception as e:
        print(f"❌ Error loading MLP: {e}")
    
    if len(all_metrics) == 0:
        print("\n❌ No models could be loaded. Check model files and architectures.")
        return
    
    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating comprehensive visualizations...")
    print(f"{'='*80}")
    
    create_comprehensive_visualizations(all_metrics)
    
    print(f"\n{'='*80}")
    print("✅ Complete! All metrics computed and visualizations generated.")
    print(f"{'='*80}")
    
    print("\n📁 Generated files:")
    print("   • complete_metrics.json (in each model directory)")
    print("   • complete_metrics_dashboard.png")
    print("   • f1_score_comparison.png")
    print("   • precision_comparison.png")
    print("   • recall_comparison.png\n")

if __name__ == "__main__":
    main()
