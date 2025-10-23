import pefile
import math

def section_entropy(data):
    if not data:
        return 0.0
    freq = [0]*256
    for b in data:
        freq[b] += 1
    ent = 0.0
    n = len(data)
    for c in freq:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent

def extract_pe_features(path):
    pe = pefile.PE(path, fast_load=True)
    pe.parse_data_directories()
    feats = {}
    oh = pe.OPTIONAL_HEADER
    feats["size_of_code"] = oh.SizeOfCode
    feats["size_of_image"] = oh.SizeOfImage
    feats["dll_char"] = oh.DllCharacteristics
    feats["num_sections"] = len(pe.sections)
    ents = []
    sizes = []
    for s in pe.sections:
        raw = s.get_data()
        ents.append(section_entropy(raw))
        sizes.append(len(raw))
    feats["sec_entropy_mean"] = sum(ents)/max(1,len(ents))
    feats["sec_entropy_std"] = float((sum((e-feats["sec_entropy_mean"])**2 for e in ents)/max(1,len(ents)))**0.5)
    feats["sec_size_mean"] = sum(sizes)/max(1,len(sizes))
    # imports
    imports = set()
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors="ignore").lower()
            imports.add(dll.split(".")[0])
    feats["num_import_dlls"] = len(imports)
    return feats, list(imports)
