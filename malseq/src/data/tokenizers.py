from collections import defaultdict

API_FAMILY_MAP = {
    # normalize common Windows API families
    "CreateFileW": "FS.Open",
    "CreateFileA": "FS.Open",
    "ReadFile": "FS.Read",
    "WriteFile": "FS.Write",
    "connect": "Net.Connect",
    "send": "Net.Send",
    "recv": "Net.Recv",
    # add more mappings
}

class APITokenizer:
    def __init__(self, unk_token="<unk>", max_vocab=4096):
        self.unk = unk_token
        self.max_vocab = max_vocab
        self.freq = defaultdict(int)
        self.stoi = {unk_token: 0}
        self.itos = [unk_token]

    def normalize(self, api_name: str) -> str:
        return API_FAMILY_MAP.get(api_name, api_name.split("Ex")[0])

    def fit(self, seqs):
        for seq in seqs:
            for raw in seq:
                tok = self.normalize(raw)
                self.freq[tok] += 1
        sorted_items = sorted(self.freq.items(), key=lambda x: -x[1])[: self.max_vocab - 1]
        for tok, _ in sorted_items:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, seq):
        return [self.stoi.get(self.normalize(t), 0) for t in seq]

class ArgBinner:
    def __init__(self, bins=(16, 64, 256, 1024, 4096)):
        self.bins = bins

    def bin_size(self, size):
        for i, b in enumerate(self.bins):
            if size <= b:
                return i + 1
        return len(self.bins) + 1

    def encode_args(self, args_seq):
        # args_seq: list of dicts with keys like {"size": int, "flags": int}
        out = []
        for a in args_seq:
            size_bin = self.bin_size(a.get("size", 0))
            flag_bin = (a.get("flags", 0) & 0xF) + 1
            out.append((size_bin, flag_bin))
        return out
