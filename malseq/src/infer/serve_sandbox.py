import torch
from ..models.multiview_model import MultiViewSeqModel
from ..models.calibration import TemperatureScaler
from ..train.early_speculate import speculate_risk
import yaml
import torch.nn.functional as F

class SandboxServer:
    def __init__(self, cfg_path, pe_dim, import_dim, device="cuda"):
        cfg = yaml.safe_load(open(cfg_path))
        self.cfg = cfg
        self.model = MultiViewSeqModel(cfg, pe_dim, import_dim).to(device)
        ckpt = torch.load("mvseq_calibrated.pt", map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.temp = TemperatureScaler()
        self.temp.load_state_dict(ckpt["temp"])
        self.model.eval()
        self.device = device
        self.ent_thresh = cfg["abstention"]["entropy_threshold"]

    @torch.no_grad()
    def predict_prefix(self, batch):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        base_logit, agg = speculate_risk(self.model, batch,
                                         beam=self.cfg["speculation"]["beam_size"],
                                         steps=self.cfg["speculation"]["steps"],
                                         aggregate=self.cfg["speculation"]["aggregate"])
        scaled = self.temp(base_logit)
        prob = torch.sigmoid(scaled)
        spec_prob = agg
        p = prob.clamp(1e-6, 1-1e-6)
        ent = -(p*torch.log(p)+(1-p)*torch.log(1-p))
        decision = torch.where(ent > self.ent_thresh, -1, (prob>=0.5).long())
        # escalate if speculative risk high but abstained
        decision = torch.where((decision==-1) & (spec_prob>0.7), torch.tensor(1, device=decision.device), decision)
        return prob.cpu(), spec_prob.cpu(), ent.cpu(), decision.cpu()
