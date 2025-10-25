import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
import torch
import random

def generate_dummy_data(n_samples=1000):
    """Generate matching PE features and API sequences with same IDs"""
    
    # Common DLLs for realistic import lists
    common_dlls = ['kernel32', 'ntdll', 'user32', 'advapi32', 'ws2_32', 
                   'shell32', 'ole32', 'oleaut32', 'wininet', 'urlmon']
    
    # Common Windows API calls
    api_calls = [
        'CreateFileW', 'CreateFileA', 'ReadFile', 'WriteFile', 'CloseHandle',
        'RegOpenKeyExW', 'RegQueryValueExW', 'RegCreateKeyExW', 'RegCloseKey',
        'VirtualAlloc', 'VirtualProtect', 'GetProcAddress', 'LoadLibraryW',
        'CreateProcessW', 'CreateThread', 'WaitForSingleObject', 'ExitProcess',
        'WSAStartup', 'socket', 'connect', 'send', 'recv', 'closesocket',
        'GetSystemInfo', 'GetTickCount', 'Sleep', 'GetCurrentProcess'
    ]
    
    pe_samples = []
    api_samples = []
    
    for i in range(n_samples):
        sample_id = f'sample_{i}'  # SAME ID for both
        
        # Generate PE features
        pe_vec = {
            'size_of_code': random.randint(1024, 1024*1024),
            'size_of_image': random.randint(8192, 2*1024*1024), 
            'dll_char': random.randint(0, 65535),
            'num_sections': random.randint(3, 12),
            'sec_entropy_mean': random.uniform(1.0, 7.8),
            'sec_entropy_std': random.uniform(0.1, 2.0),
            'sec_size_mean': random.randint(1024, 512*1024),
            'num_import_dlls': random.randint(2, 20)
        }
        
        n_imports = random.randint(3, 8)
        imports = random.sample(common_dlls, min(n_imports, len(common_dlls)))
        
        pe_samples.append({
            'id': sample_id,
            'pe_vec': pe_vec,
            'imports': imports
        })
        
        # Generate API sequence with SAME ID
        seq_len = random.randint(50, 500)
        apis = []
        args = []
        
        for j in range(seq_len):
            api = random.choice(api_calls)
            apis.append(api)
            size = random.randint(0, 65536) if random.random() > 0.3 else 0
            flags = random.randint(0, 15) if random.random() > 0.5 else 0
            args.append({'size': size, 'flags': flags})
        
        # 60% malware, 40% benign
        label = 1 if random.random() < 0.6 else 0
        
        api_samples.append({
            'id': sample_id,  # SAME ID
            'apis': apis,
            'args': args,
            'label': label
        })
    
    return pe_samples, api_samples

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    
    print("Generating dummy data with matching IDs...")
    pe_samples, api_samples = generate_dummy_data(1000)
    
    # Save PE features
    torch.save(pe_samples, 'data/static_raw.pt')
    print(f"✓ Created data/static_raw.pt with {len(pe_samples)} samples")
    
    # Save API sequences
    json.dump(api_samples, open('data/dynamic_raw.json', 'w'))
    print(f"✓ Created data/dynamic_raw.json with {len(api_samples)} samples")
    
    # Verify IDs match
    pe_ids = set(s['id'] for s in pe_samples)
    api_ids = set(s['id'] for s in api_samples)
    assert pe_ids == api_ids, "ID mismatch!"
    print(f"✓ Verified all {len(pe_ids)} IDs match between static and dynamic")
    
    print("\nDone! Now run: python scripts/tokenize_dynamic.py")
