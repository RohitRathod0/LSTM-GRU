import torch
from src.models.multiview_model import MultiViewSeqModel
from src.train.early_speculate import speculate_risk
import yaml

def test_speculation_shapes():
    cfg = yaml.safe_load(open("configs/mv_gru_default.yaml"))
    model = MultiViewSeqModel(cfg, pe_dim=64, import_dim=512)
    B, T = 2, 40
    batch = {
        "pe_vec": torch.randn(B, 64),
        "import_vec": torch.randn(B, 512),
        "tok_ids": torch.randint(0, cfg["dynamic"]["vocab_size"], (B, T)),
        "arg_ids": torch.randint(0, cfg["dynamic"]["arg_vocab_size"], (B, T)),
        "burst": torch.ones(B, T, 1),
        "pos": torch.arange(T).float().unsqueeze(0).repeat(B,1).unsqueeze(-1),
        "label": torch.randint(0,2,(B,)),
        "next_token": torch.randint(0, cfg["dynamic"]["vocab_size"], (B,))
    }
    base, agg = speculate_risk(model, batch, beam=2, steps=4)
    assert base.shape[0] == B and agg.shape[0] == B
