import torch

class ReplayBuffer:
    def __init__(self, size=5000):
        self.size = size
        self.buf = []

    def add(self, sample):
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(sample)

    def loader(self, batch_size, device):
        def collate(batch):
            out = {}
            for k in batch[0]:
                if isinstance(batch[0][k], torch.Tensor):
                    out[k] = torch.stack([b[k] for b in batch]).to(device)
                else:
                    out[k] = [b[k] for b in batch]
            return out
        for i in range(0, len(self.buf), batch_size):
            yield collate(self.buf[i:i+batch_size])
