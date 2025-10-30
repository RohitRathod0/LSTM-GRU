import pandas as pd
import numpy as np

# Column names for NSL-KDD
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'difficulty'
]

print("="*60)
print("NSL-KDD Dataset Preprocessing")
print("="*60)

# Load data
print("\nLoading NSL-KDD data...")
train = pd.read_csv('data/KDDTrain+.txt', names=columns, header=None)
test = pd.read_csv('data/KDDTest+.txt', names=columns, header=None)

print(f"✓ Train shape: {train.shape}")
print(f"✓ Test shape: {test.shape}")

# Drop difficulty column
train = train.drop('difficulty', axis=1)
test = test.drop('difficulty', axis=1)

# Clean attack labels (remove trailing dots)
train['attack'] = train['attack'].str.strip().str.rstrip('.')
test['attack'] = test['attack'].str.strip().str.rstrip('.')

print(f"\n✓ Unique attacks in training: {train['attack'].nunique()}")
print(f"  Samples: {train['attack'].value_counts().head(10)}\n")

# Create binary label: 0=Normal, 1=Attack (Malware)
train['label'] = (train['attack'] != 'normal').astype(int)
test['label'] = (test['attack'] != 'normal').astype(int)

print("Label distribution (Binary Classification):")
print(f"  Train - Normal: {(train['label']==0).sum():,}, Attack: {(train['label']==1).sum():,}")
print(f"  Test  - Normal: {(test['label']==0).sum():,}, Attack: {(test['label']==1).sum():,}")

# One-hot encode categorical features
print("\n✓ One-hot encoding categorical features...")
categorical = ['protocol_type', 'service', 'flag']
train = pd.get_dummies(train, columns=categorical)
test = pd.get_dummies(test, columns=categorical)

# Align test columns with train (handle unseen categories)
missing_cols = set(train.columns) - set(test.columns)
for col in missing_cols:
    test[col] = 0
test = test[train.columns]

# Drop original attack column
train = train.drop('attack', axis=1)
test = test.drop('attack', axis=1)

# Save processed data
print("\n✓ Saving processed datasets...")
train.to_csv('data/nslkdd_train.csv', index=False)
test.to_csv('data/nslkdd_test.csv', index=False)

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nSaved files:")
print(f"  📄 data/nslkdd_train.csv - {len(train):,} samples, {len(train.columns)} features")
print(f"  📄 data/nslkdd_test.csv  - {len(test):,} samples, {len(test.columns)} features")

print(f"\n📊 Dataset Statistics:")
print(f"  Features: {len(train.columns)-1} (+ 1 label)")
print(f"  Classes: Binary (Normal vs Attack)")
print(f"  Train samples: {len(train):,}")
print(f"  Test samples:  {len(test):,}")

print(f"\n🚀 Ready to train models!")
print(f"\nNext commands:")
print(f"  # Train LSTM")
print(f"  python train_flows.py --csv data/nslkdd_train.csv --label label --model lstm --epochs 10 --batch 512 --outdir runs/nslkdd_lstm")
print(f"\n  # Train GRU")
print(f"  python train_flows.py --csv data/nslkdd_train.csv --label label --model gru --epochs 10 --batch 512 --outdir runs/nslkdd_gru")
print(f"\n  # Train MLP")
print(f"  python train_flows.py --csv data/nslkdd_train.csv --label label --model mlp --epochs 10 --batch 512 --outdir runs/nslkdd_mlp")
print("="*60)
