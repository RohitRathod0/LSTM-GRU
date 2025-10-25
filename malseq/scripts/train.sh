#!/usr/bin/env bash
set -e
python -c "
import sys, os
sys.path.insert(0, '.')

import torch
import yaml

# Load a sample to determine dimensions
sample = torch.load('data/train.pt')[0]
pe_dim = sample['pe_vec'].shape[0]
import_dim = sample['import_vec'].shape[0]

print(f'Detected PE dim: {pe_dim}, Import dim: {import_dim}')

from src.train.train_multiview import train
train('configs/mv_gru_default.yaml', 
     torch.load('data/train.pt'), 
     torch.load('data/val.pt'), 
     pe_dim=pe_dim, 
     import_dim=import_dim, 
     device='cuda' if torch.cuda.is_available() else 'cpu')
"
