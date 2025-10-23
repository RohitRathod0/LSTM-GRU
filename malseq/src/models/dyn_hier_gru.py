import torch
import torch.nn as nn

class WindowEncoder(nn.Module):
    def __init__(self, tok_dim, arg_dim, hidden, bidir=True, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(tok_dim + arg_dim + 2, hidden, num_layers=1,
                          batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, win_embed, burst_feat, pos_feat):
        # win_embed: [B, W, tok_dim+arg_dim]
        x = torch.cat([win_embed, burst_feat, pos_feat], dim=-1)
        out, _ = self.gru(x)
        out = self.dropout(out)
        return out.mean(dim=1)  # window summary

class HierGRU(nn.Module):
    def __init__(self, tok_vocab, arg_vocab, tok_dim=128, arg_dim=16,
                 win_len=32, win_hidden=128, seq_hidden=192,
                 bidir_windows=True, dropout=0.2):
        super().__init__()
        self.tok_emb = nn.Embedding(tok_vocab, tok_dim)
        self.arg_emb = nn.Embedding(arg_vocab, arg_dim)
        self.win_len = win_len
        self.win_enc = WindowEncoder(tok_dim, arg_dim, win_hidden, bidir_windows, dropout)
        self.seq_gru = nn.GRU(self.win_enc.out_dim, seq_hidden,
                              batch_first=True, bidirectional=False)
        self.seq_hidden = seq_hidden

    def _chunk(self, x, L):
        # x: [B, T, D] -> [B, Nw, L, D]
        B, T, D = x.shape
        pad = (L - (T % L)) % L
        if pad:
            pad_t = torch.zeros(B, pad, D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_t], dim=1)
        Nw = x.shape[1] // L
        return x.view(B, Nw, L, D), Nw

    def forward(self, tok_ids, arg_ids, burst, pos):
        # tok_ids/arg_ids: [B, T]; burst/pos: [B, T, 1]
        tok_e = self.tok_emb(tok_ids)
        arg_e = self.arg_emb(arg_ids)
        inp = torch.cat([tok_e, arg_e], dim=-1)
        chunked, Nw = self._chunk(inp, self.win_len)
        burst_c, _ = self._chunk(burst, self.win_len)
        pos_c, _ = self._chunk(pos, self.win_len)
        B, Nw, L, D = chunked.shape
        wins = []
        for i in range(Nw):
            w = chunked[:, i]
            b = burst_c[:, i]
            p = pos_c[:, i]
            wins.append(self.win_enc(w, b, p))
        win_seq = torch.stack(wins, dim=1)  # [B, Nw, H]
        seq_out, _ = self.seq_gru(win_seq)
        return seq_out[:, -1, :]  # final sequence representation
