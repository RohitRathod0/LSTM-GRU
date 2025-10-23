import torch
import torch.nn as nn
from .static_encoders import StaticEncoder
from .dyn_hier_gru import HierGRU
from .fusion_gate import FusionGate
from .heads import ClassifierHead, NextTokenHead

class MultiViewSeqModel(nn.Module):
    def __init__(self, cfg, pe_dim, import_dim):
        super().__init__()
        self.static_enc = StaticEncoder(pe_dim, import_dim, hid=cfg["static"]["hidden_dim"])
        self.dyn_enc = HierGRU(
            tok_vocab=cfg["dynamic"]["vocab_size"],
            arg_vocab=cfg["dynamic"]["arg_vocab_size"],
            tok_dim=cfg["dynamic"]["token_dim"],
            arg_dim=cfg["dynamic"]["arg_dim"],
            win_len=cfg["dynamic"]["window_len"],
            win_hidden=cfg["dynamic"]["win_bi_gru_dim"],
            seq_hidden=cfg["dynamic"]["seq_gru_dim"],
            bidir_windows=cfg["dynamic"]["bidirectional_windows"],
            dropout=cfg["dynamic"]["dropout"],
        )
        s_dim = cfg["static"]["hidden_dim"]
        d_dim = cfg["dynamic"]["seq_gru_dim"]
        self.gate = FusionGate(s_dim, hid=cfg["fusion"]["gate_hidden"], dropout=cfg["fusion"]["gate_dropout"])
        self.cls_static = ClassifierHead(s_dim, hid=cfg["heads"]["classifier_hidden"], out=1)
        self.cls_dyn = ClassifierHead(d_dim, hid=cfg["heads"]["classifier_hidden"], out=1)
        self.next_tok = NextTokenHead(d_dim, cfg["dynamic"]["vocab_size"])

    def forward(self, batch):
        # batch: dict with pe_vec, import_vec, tok_ids, arg_ids, burst, pos
        s_vec = self.static_enc(batch["pe_vec"], batch["import_vec"])
        d_vec = self.dyn_enc(batch["tok_ids"], batch["arg_ids"], batch["burst"], batch["pos"])
        gate = self.gate(s_vec)  # [B,1]
        logit_static = self.cls_static(s_vec)
        logit_dyn = self.cls_dyn(d_vec)
        logit = gate * logit_dyn + (1 - gate) * logit_static
        next_logits = self.next_tok(d_vec)
        return {"logit": logit.squeeze(-1), "logit_static": logit_static.squeeze(-1),
                "logit_dyn": logit_dyn.squeeze(-1), "gate": gate.squeeze(-1),
                "next_logits": next_logits}
