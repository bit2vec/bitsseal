#!/usr/bin/env python3
"""Load HuggingFace HyenaDNA-tiny-1k-seqlen-hf and save its state_dict
in our checkpoint format (with key 'model_state_dict') so it can be used
as --init_ckpt for hyena_wikitext_to_dna.py.

This is what was loaded for the HyenaDNA-tiny baseline (test 0.873)."""
import argparse, os, torch
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', default='LongSafari/hyenadna-tiny-1k-seqlen-hf')
    p.add_argument('--out', required=True,
                   help='Output .pt file (e.g. results/.../hf_hg38_pretrain.pt)')
    args = p.parse_args()

    print(f'loading {args.model_id}', flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id,
                                                   trust_remote_code=True)
    sd = model.state_dict()
    out = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in sd.items()},
           'source': args.model_id, 'note': 'HF HyenaDNA-tiny pretrained on hg38'}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(out, args.out)
    print(f'wrote {args.out}', flush=True)
    print(f'  {len(sd)} tensors, '
          f'{sum(v.numel() for v in sd.values()):,} params', flush=True)
    # Sanity print a few keys
    for k in list(sd.keys())[:6]:
        print(f'  {k}: {tuple(sd[k].shape)}')


if __name__ == '__main__':
    main()
