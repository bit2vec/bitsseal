#!/usr/bin/env python3
"""EXACT original from-scratch recipe (bits_lm_scratch.py) + intermediate checkpoints every 100M
tokens, so we get the full 100M->500M trajectory for both arms. Training dynamics are identical to
the original (only checkpoint saving is added).
Camera-ready additions (2026-11-05): transformers 5 warmup_steps (= ceil(0.02*max_steps), what
warmup_ratio=0.02 computed before), multi-GPU token accounting via WORLD_SIZE, optional FSDP
sharding (same optimizer step, states sharded across ranks), keep all checkpoints."""
import argparse, sys, math, os, torch
sys.path.insert(0, '/root')
from bits_llm import swap_in_bits, get_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
import subprocess


class Bf16Save(TrainerCallback):
    """Disk only: after each save, rewrite the fp32 safetensors of that checkpoint to bf16 in a detached
    process (30G -> 15G per 7B checkpoint). Training dynamics untouched."""
    def on_save(self, args, state, control, **kw):
        if state.is_world_process_zero:
            d = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
            subprocess.Popen([sys.executable, '/root/to_bf16.py', d], stdout=open(d + '/bf16.log', 'w'), stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['bits', 'standard'])
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B')
    ap.add_argument('--tokens', type=int, default=500_000_000)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--nbits', type=int, default=256)
    ap.add_argument('--W', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fsdp', action='store_true')
    ap.add_argument('--save_tokens', type=int, default=100_000_000)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    if not a.fsdp:
        model = model.cuda()
    if a.arm == 'bits':
        swap_in_bits(model, a.nbits, a.W, a.seed)
        if not a.fsdp:
            model = model.cuda()
    nparam = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for p in model.get_input_embeddings().parameters())
    print(f"[{a.arm} s{a.seed}] total={nparam/1e6:.0f}M  embed={emb/1e6:.2f}M", flush=True)
    train_ds = get_dataset(tok, a.max_len, a.tokens, 'train')
    world = int(os.environ.get('WORLD_SIZE', 1))
    tps = a.max_len * a.bs * a.grad_accum * world   # tokens per optimizer step across all ranks
    max_steps = max(1, a.tokens // tps)
    save_every = max(1, a.save_tokens // tps)       # checkpoint every 100M tokens
    extra = {}
    if a.fsdp:
        extra = dict(fsdp='full_shard auto_wrap',
                     fsdp_config={'transformer_layer_cls_to_wrap': ['Qwen2DecoderLayer']})
    targs = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, max_steps=max_steps, logging_steps=10, bf16=True,
        save_strategy='steps', save_steps=save_every, save_total_limit=None, save_only_model=True,
        optim='adamw_torch', gradient_checkpointing=True, report_to=['tensorboard'], dataloader_num_workers=2,
        warmup_steps=math.ceil(0.02 * max_steps), lr_scheduler_type='cosine', max_grad_norm=1.0, seed=a.seed,
        **extra)
    def collate(b):
        return {'input_ids': torch.tensor([x['input_ids'] for x in b]),
                'labels': torch.tensor([x['labels'] for x in b])}
    tr = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collate)
    tr.add_callback(Bf16Save())
    print(f"[{a.arm} s{a.seed}] world={world} tokens/step={tps} max_steps={max_steps} save_every={save_every}", flush=True)
    tr.train()
    tr.save_model(a.out + '/final')
    final = tr.state.log_history[-1].get('loss', float('nan')) if tr.state.log_history else float('nan')
    print(f"[{a.arm} s{a.seed}] max_steps={max_steps} save_every={save_every} DONE  final_loss~={final}", flush=True)


if __name__ == '__main__':
    main()
