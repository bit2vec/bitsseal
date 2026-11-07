import sys, os, glob, torch
from safetensors.torch import load_file, save_file
d = sys.argv[1]
for p in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
    sd = load_file(p)
    n = sum(v.numel() for v in sd.values())
    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    save_file(sd, p + '.tmp', metadata={'format': 'pt'})
    os.replace(p + '.tmp', p)
    print('bf16', p, n, flush=True)
print('DONE', flush=True)
