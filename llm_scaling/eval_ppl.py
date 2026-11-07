#!/usr/bin/env python3
"""Held-out perplexity eval for Bits vs Standard checkpoints (supports encoder variants).
Same held-out set, same metric exp(mean per-token CE). For Bits variants, pass the SAME
--enc_type/--ffn_depth/--W/--nbits used at training so the module structure matches for loading."""
import argparse, glob, os, math, sys, torch
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch.nn.functional as F
sys.path.insert(0, '/root')
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file


def build(a):
    cfg = AutoConfig.from_pretrained(a.ckpt)
    if a.arm == 'standard':
        return AutoModelForCausalLM.from_pretrained(
            a.ckpt, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    from bits_llm import swap_in_bits
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    swap_in_bits(model, a.nbits, a.W, a.seed, enc_type=a.enc_type, ffn_depth=a.ffn_depth, out_scale=0.0)
    sd = {}
    for f in glob.glob(os.path.join(a.ckpt, '*.safetensors')):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    enc_keys = [k for k in sd if 'embed_tokens' in k or '.ffn.' in k or 'in_proj' in k or 'out_proj' in k or 'tfenc' in k]
    print(f"[load] tensors={len(sd)} enc_keys={len(enc_keys)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if len(enc_keys) == 0:
        print("[load][FATAL] no encoder weights in checkpoint", flush=True); sys.exit(3)
    return model.cuda().eval()


@torch.no_grad()
def evaluate(model, seqs, bs):
    tot_loss, tot_tok = 0.0, 0
    for i in range(0, len(seqs), bs):
        batch = torch.tensor(seqs[i:i+bs]).cuda()
        logits = model(input_ids=batch).logits[:, :-1, :].float()
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction='sum')
        tot_loss += loss.item(); tot_tok += labels.numel()
    ml = tot_loss / tot_tok
    return ml, math.exp(ml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['bits', 'standard'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='/root/heldout_c4val_4000x512.pt')
    ap.add_argument('--nbits', type=int, default=256)
    ap.add_argument('--W', type=int, default=512)
    ap.add_argument('--enc_type', default='mlp', choices=['mlp', 'tfenc'])
    ap.add_argument('--ffn_depth', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_seq', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=8)
    a = ap.parse_args()
    model = build(a)
    seqs = torch.load(a.data)[:a.n_seq]
    ml, ppl = evaluate(model, seqs, a.bs)
    print(f"RESULT arm={a.arm} ckpt={a.ckpt} enc={a.enc_type} depth={a.ffn_depth} W={a.W} nbits={a.nbits} "
          f"tokens={len(seqs)*len(seqs[0])} eval_loss={ml:.4f} ppl={ppl:.2f}", flush=True)


if __name__ == '__main__':
    main()
