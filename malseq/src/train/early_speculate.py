import torch
import torch.nn.functional as F

def speculate_risk(model, batch, beam=3, steps=8, aggregate="meanmax"):
    # batch has prefix tok_ids, arg_ids, burst, pos
    out = model(batch)
    base_logit = out["logit"]
    B = base_logit.shape[0]
    risks = []
    # Use next token distribution from current d_vec representation
    next_logits = out["next_logits"]  # [B, V]
    probs = F.softmax(next_logits, dim=-1)
    topk = torch.topk(probs, beam, dim=-1).indices  # [B, beam]
    for b in range(beam):
        # clone batch and extend with a constant arg-bin and position increments
        cloned = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        tok_extend = topk[:, b].unsqueeze(1)
        for _ in range(steps):
            cloned["tok_ids"] = torch.cat([cloned["tok_ids"], tok_extend], dim=1)
            cloned["arg_ids"] = torch.cat([cloned["arg_ids"], torch.ones_like(tok_extend)], dim=1)
            # simplistic pos/burst increments
            last_pos = cloned["pos"][:, -1:]
            cloned["pos"] = torch.cat([cloned["pos"], last_pos + 1.0], dim=1)
            cloned["burst"] = torch.cat([cloned["burst"], cloned["burst"][:, -1:]], dim=1)
        fut = model(cloned)
        risks.append(torch.sigmoid(fut["logit"]))
    R = torch.stack(risks, dim=-1)  # [B, beam]
    if aggregate == "meanmax":
        agg = 0.5 * R.mean(dim=-1) + 0.5 * R.max(dim=-1).values
    elif aggregate == "max":
        agg = R.max(dim=-1).values
    else:
        agg = R.mean(dim=-1)
    return base_logit, agg
