import torch
import torch.nn.functional as F
from ..models.multiview_model import MultiViewSeqModel
from ..models.calibration import TemperatureScaler
import yaml

class EdgeServer:
    def __init__(self, cfg_path, pe_dim, import_dim, device="cpu"):
        cfg = yaml.safe_load(open(cfg_path))
        self.model = MultiViewSeqModel(cfg, pe_dim, import_dim).to(device)
        ckpt = torch.load("mvseq_calibrated.pt", map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.temp = TemperatureScaler()
        self.temp.load_state_dict(ckpt["temp"])
        self.model.eval()
        self.device = device
        self.ent_thresh = cfg["abstention"]["entropy_threshold"]

    @torch.no_grad()
    def predict(self, batch):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        out = self.model(batch)
        scaled = self.temp(out["logit"])
        prob = torch.sigmoid(scaled)
        # entropy-based abstention
        p = prob.clamp(1e-6, 1-1e-6)
        ent = -(p*torch.log(p)+(1-p)*torch.log(1-p))
        decision = torch.where(ent > self.ent_thresh, -1, (prob>=0.5).long())  # -1 means abstain
        return prob.cpu(), ent.cpu(), decision.cpu()
