import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from train_flows import LSTM_Classifier, GRU_Classifier, MLP_Classifier, load_data, batch_iter_mlp, batch_iter_seq

def evaluate_model(model_type, checkpoint_dir, test_csv, label_col="label", seq_len=20):
    """Evaluate a trained model on test set"""
    
    # Load test data
    _, _, _, _, Xte, yte, _, num_classes = load_data(test_csv, label_col=label_col)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_features = Xte.shape[1]
    
    # Load model
    if model_type == "lstm":
        features_per_step = total_features // seq_len
        model = LSTM_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        batch_fn = lambda X, y, b, s: batch_iter_seq(X, y, b, seq_len, s)
    elif model_type == "gru":
        features_per_step = total_features // seq_len
        model = GRU_Classifier(features_per_step, num_classes, hidden=128, num_layers=2).to(device)
        batch_fn = lambda X, y, b, s: batch_iter_seq(X, y, b, seq_len, s)
    elif model_type == "mlp":
        model = MLP_Classifier(total_features, num_classes, width=256).to(device)
        batch_fn = batch_iter_mlp
    
    # Load weights
    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Get predictions
    vs, ps = [], []
    with torch.no_grad():
        for xb, yb in batch_fn(Xte, yte, 512, False):
            xb = xb.to(device)
            prob = F.softmax(model(xb), dim=-1).cpu().numpy()
            ps.append(prob)
            vs.append(yb.numpy())
    
    ps = np.vstack(ps)
    vs = np.concatenate(vs)
    y_pred = ps.argmax(1)
    
    # Compute metrics
    acc = accuracy_score(vs, y_pred)
    results = {
        "model": model_type,
        "accuracy": acc,
        "num_samples": len(vs)
    }
    
    if num_classes == 2:
        auc = roc_auc_score(vs, ps[:, 1])
        results["roc_auc"] = auc
    
    # Confusion matrix
    cm = confusion_matrix(vs, y_pred)
    
    return results, cm, vs, y_pred, ps

def main():
    test_csv = "../data/nslkdd_train.csv"  # Using train CSV but it will split internally
    models = ["lstm", "gru", "mlp"]
    
    print("\n" + "="*70)
    print(" "*20 + "MODEL EVALUATION RESULTS")
    print("="*70 + "\n")
    
    all_results = []
    
    for model_type in models:
        checkpoint_dir = f"runs/nslkdd_{model_type}"
        
        print(f"Evaluating {model_type.upper()}...")
        results, cm, y_true, y_pred, probs = evaluate_model(model_type, checkpoint_dir, test_csv)
        
        if results:
            all_results.append(results)
            
            print(f"\n{'='*70}")
            print(f"{model_type.upper()} Results:")
            print(f"{'='*70}")
            print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            if 'roc_auc' in results:
                print(f"  ROC-AUC:  {results['roc_auc']:.4f}")
            print(f"  Samples:  {results['num_samples']:,}")
            
            # Classification report
            print(f"\n{model_type.upper()} Classification Report:")
            target_names = ['Normal', 'Attack']
            print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))
            
            # Save confusion matrix
            cm_path = os.path.join(checkpoint_dir, "confusion_matrix.csv")
            np.savetxt(cm_path, cm, fmt='%d', delimiter=',')
            print(f"✓ Saved confusion matrix: {cm_path}\n")
    
    # Comparison table
    print("\n" + "="*70)
    print(" "*25 + "COMPARISON")
    print("="*70)
    print(f"{'Model':<10} | {'Accuracy':<12} | {'ROC-AUC':<10} | {'Samples':<10}")
    print("-"*70)
    
    for r in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
        auc_str = f"{r['roc_auc']:.4f}" if 'roc_auc' in r else "N/A"
        print(f"{r['model']:<10} | {r['accuracy']:.4f} ({r['accuracy']*100:5.2f}%) | {auc_str:<10} | {r['num_samples']:<10,}")
    
    print("="*70)
    print("✅ Evaluation complete!\n")

if __name__ == "__main__":
    main()
