import sys
sys.path.insert(0, '.')

import torch
import yaml
from src.train.train_multiview import train

# Auto-detect dimensions from data
sample = torch.load('data/train.pt')[0]
pe_dim = sample['pe_vec'].shape[0]
import_dim = sample['import_vec'].shape[0]

print(f'Starting training with PE_dim={pe_dim}, Import_dim={import_dim}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

train('configs/mv_gru_default.yaml', 
     torch.load('data/train.pt'), 
     torch.load('data/val.pt'), 
     pe_dim, 
     import_dim, 
     device)

print('\n✓ Training complete! Model saved as mvseq_calibrated.pt')
