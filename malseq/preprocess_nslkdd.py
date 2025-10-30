"""
preprocess_nslkdd.py - Convert NSL-KDD txt to preprocessed npz
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# NSL-KDD column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def preprocess_data(input_file, output_file):
    """Load, preprocess, and save NSL-KDD data"""
    
    print(f"Loading {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file, names=columns, header=None)
    
    print(f"Loaded {len(df)} samples")
    
    # Drop difficulty level
    df = df.drop('difficulty', axis=1)
    
    # Convert labels to binary (normal vs attack)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Separate features and labels
    y = df['label'].values
    X = df.drop('label', axis=1)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    X = X.values.astype(np.float32)
    y = y.astype(np.int64)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Final shape: X={X.shape}, y={y.shape}")
    
    # Save as npz
    np.savez(output_file, X=X, y=y)
    print(f"✓ Saved to {output_file}\n")

def main():
    print("\n" + "="*80)
    print(" "*25 + "NSL-KDD PREPROCESSING")
    print("="*80 + "\n")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Preprocess train set
    if Path("data/KDDTrain+.txt").exists():
        preprocess_data("data/KDDTrain+.txt", "data/nslkdd_train_preprocessed.npz")
    else:
        print("⚠ Warning: data/KDDTrain+.txt not found, skipping")
    
    # Preprocess test set
    if Path("data/KDDTest+.txt").exists():
        preprocess_data("data/KDDTest+.txt", "data/nslkdd_test_preprocessed.npz")
    else:
        print("⚠ Warning: data/KDDTest+.txt not found, skipping")
    
    print("="*80)
    print("✅ Preprocessing complete!")
    print("="*80)

if __name__ == "__main__":
    main()
