import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from train_flows import LSTM_Classifier, GRU_Classifier, MLP_Classifier, load_data
import matplotlib.pyplot as plt

def evaluate_prefix_sequential(model, X_test, y_test, prefix_features, device, seq_len=20, features_per_step=6):
    """Evaluate sequential model (LSTM/GRU) with only first k features"""
    
    # Adjust prefix to be multiple of features_per_step
    adjusted_prefix = (prefix_features // features_per_step) * features_per_step
    
    if adjusted_prefix < features_per_step:
        adjusted_prefix = features_per_step
    
    X_prefix = X_test[:, :adjusted_prefix]
    actual_seq_len = adjusted_prefix // features_per_step
    X_prefix = X_prefix.reshape(-1, actual_seq_len, features_per_step)
    
    model.eval()
    vs, ps = [], []
    
    with torch.no_grad():
        for i in range(0, len(X_prefix), 512):
            xb = torch.from_numpy(X_prefix[i:i+512]).to(device)
            yb = y_test[i:i+512]
            prob = F.softmax(model(xb), dim=-1).cpu().numpy()
            ps.append(prob)
            vs.append(yb)
    
    ps = np.vstack(ps)
    vs = np.concatenate(vs)
    
    acc = accuracy_score(vs, ps.argmax(1))
    auc = roc_auc_score(vs, ps[:, 1]) if ps.shape[1] == 2 else 0.0
    
    return acc, auc

def evaluate_prefix_mlp(model, X_test, y_test, prefix_features, device, total_features):
    """Evaluate MLP with padding for partial features"""
    
    # Pad prefix features with zeros to match model input size
    X_prefix = X_test[:, :prefix_features]
    padding = np.zeros((X_prefix.shape[0], total_features - prefix_features), dtype=np.float32)
    X_padded = np.concatenate([X_prefix, padding], axis=1)
    
    model.eval()
    vs, ps = [], []
    
    with torch.no_grad():
        for i in range(0, len(X_padded), 512):
            xb = torch.from_numpy(X_padded[i:i+512]).to(device)
            yb = y_test[i:i+512]
            prob = F.softmax(model(xb), dim=-1).cpu().numpy()
            ps.append(prob)
            vs.append(yb)
    
    ps = np.vstack(ps)
    vs = np.concatenate(vs)
    
    acc = accuracy_score(vs, ps.argmax(1))
    auc = roc_auc_score(vs, ps[:, 1]) if ps.shape[1] == 2 else 0.0
    
    return acc, auc

def main():
    test_csv = "../data/nslkdd_train.csv"
    models_info = [
        ("lstm", "runs/nslkdd_lstm", "sequential"),
        ("gru", "runs/nslkdd_gru", "sequential"),
        ("mlp", "runs/nslkdd_mlp", "tabular")
    ]
    
    # Load test data
    _, _, _, _, Xte, yte, _, num_classes = load_data(test_csv, label_col="label")
    total_features = Xte.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define prefix lengths to test
    prefix_ratios = [0.25, 0.5, 0.75, 1.0]
    prefix_features = [int(total_features * r) for r in prefix_ratios]
    
    print("\n" + "="*70)
    print(" "*20 + "EARLY DETECTION ANALYSIS")
    print("="*70)
    print(f"Total features: {total_features}")
    print(f"Testing prefix lengths: {prefix_features}")
    print(f"Prefix ratios: {[f'{r*100:.0f}%' for r in prefix_ratios]}\n")
    
    results_all = {}
    
    for model_type, checkpoint_dir, model_class in models_info:
        print(f"\nEvaluating {model_type.upper()} with progressive feature sets...")
        
        # Load model
        seq_len = 20
        features_per_step = total_features // seq_len
        
        if model_type == "lstm":
            model = LSTM_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        elif model_type == "gru":
            model = GRU_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        elif model_type == "mlp":
            model = MLP_Classifier(total_features, num_classes, width=256).to(device)
        
        checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
        if not os.path.exists(checkpoint_path):
            print(f"  ❌ Checkpoint not found: {checkpoint_path}")
            continue
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        results = []
        for k in prefix_features:
            try:
                if model_class == "sequential":
                    acc, auc = evaluate_prefix_sequential(model, Xte, yte, k, device, seq_len, features_per_step)
                else:  # MLP
                    acc, auc = evaluate_prefix_mlp(model, Xte, yte, k, device, total_features)
                
                results.append({"features": k, "ratio": k/total_features, "accuracy": acc, "auc": auc})
                print(f"  Features: {k:3d} ({k/total_features*100:5.1f}%) | Acc: {acc:.4f} | AUC: {auc:.4f}")
            except Exception as e:
                print(f"  ⚠️  Error at k={k}: {e}")
        
        results_all[model_type] = results
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for model_type, results in results_all.items():
        if len(results) > 0:
            ratios = [r['ratio'] * 100 for r in results]
            accs = [r['accuracy'] for r in results]
            plt.plot(ratios, accs, marker='o', label=model_type.upper(), linewidth=2)
    plt.xlabel('Feature Availability (%)', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.title('Early Detection: Accuracy vs. Feature Completeness', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    # AUC plot
    plt.subplot(1, 2, 2)
    for model_type, results in results_all.items():
        if len(results) > 0:
            ratios = [r['ratio'] * 100 for r in results]
            aucs = [r['auc'] for r in results]
            plt.plot(ratios, aucs, marker='s', label=model_type.upper(), linewidth=2)
    plt.xlabel('Feature Availability (%)', fontsize=11)
    plt.ylabel('ROC-AUC', fontsize=11)
    plt.title('Early Detection: ROC-AUC vs. Feature Completeness', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.0])
    
    plt.tight_layout()
    plot_path = "runs/early_detection_comparison.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\n✓ Saved plot: {plot_path}")
    
    # Summary table
    print("\n" + "="*70)
    print(" "*20 + "EARLY DETECTION SUMMARY")
    print("="*70)
    print(f"{'Model':<10} | {'25% Features':<15} | {'50% Features':<15} | {'75% Features':<15} | {'100% Features':<15}")
    print("-"*90)
    
    for model_type, results in results_all.items():
        if len(results) >= 4:
            accs = [f"{r['accuracy']:.4f}" for r in results]
            print(f"{model_type.upper():<10} | {accs[0]:<15} | {accs[1]:<15} | {accs[2]:<15} | {accs[3]:<15}")
        elif len(results) > 0:
            print(f"{model_type.upper():<10} | Incomplete results ({len(results)} evaluations)")
    
    print("="*90)
    
    # Key insights
    print("\n📊 Key Findings:")
    for model_type, results in results_all.items():
        if len(results) >= 2:
            acc_50 = results[1]['accuracy']
            acc_100 = results[-1]['accuracy']
            print(f"  - {model_type.upper()}: {acc_50*100:.1f}% accuracy at 50% features ({acc_50/acc_100*100:.1f}% of full performance)")
    
    print("\n✅ Early detection analysis complete!\n")

if __name__ == "__main__":
    main()
