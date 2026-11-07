"""
Bits-for-LLM encoder (extended): token id -> fixed random n-bit code -> encoder -> hidden.
Backward compatible: defaults (enc_type='mlp', ffn_depth=2, out_scale=0) reproduce the
original 2-layer MLP exactly. New encoder-only tuning knobs for the 7B sweep:
  - ffn_depth : number of Linear layers in the MLP (2 = original, 3 = deeper)
  - out_scale : if >0, rescale the final projection so output std ~= out_scale (init-scale match)
  - enc_type='tfenc' : small Transformer encoder over the bit-code (TF-Enc-H4-FFN512 replica)
"""
import torch, torch.nn as nn


class BitsEmbedding(nn.Module):
    def __init__(self, vocab, hidden, nbits=64, W=256, seed=0, dtype=torch.bfloat16,
                 enc_type='mlp', ffn_depth=2, out_scale=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # plain attribute (NOT a buffer) so FSDP never shards it
        self._codes = torch.randint(0, 2, (vocab, nbits), generator=g).to(dtype)
        self.vocab, self.hidden, self.nbits, self.W = vocab, hidden, nbits, W
        self.enc_type, self.ffn_depth = enc_type, ffn_depth
        self.padding_idx = None
        if enc_type == 'tfenc':
            self.tok_bits = 16
            assert nbits % self.tok_bits == 0, "nbits must be divisible by 16 for tfenc"
            self.seq = nbits // self.tok_bits
            self.in_proj = nn.Linear(self.tok_bits, W)
            self.pos = nn.Parameter(torch.zeros(self.seq, W))
            layer = nn.TransformerEncoderLayer(d_model=W, nhead=4, dim_feedforward=512,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.tfenc = nn.TransformerEncoder(layer, num_layers=2)
            self.out_proj = nn.Linear(W, hidden)
            self.to(dtype)
        else:  # mlp with ffn_depth linear layers (>=2)
            layers = [nn.Linear(nbits, W), nn.GELU()]
            for _ in range(max(0, ffn_depth - 2)):
                layers += [nn.Linear(W, W), nn.GELU()]
            layers += [nn.Linear(W, hidden)]
            self.ffn = nn.Sequential(*layers).to(dtype)
        if out_scale and out_scale > 0:
            self._match_out_scale(out_scale)

    def _final(self):
        return self.out_proj if self.enc_type == 'tfenc' else self.ffn[-1]

    def _match_out_scale(self, target_std):
        # empirically rescale the final (linear) layer so the OUTPUT embedding std ~= target_std,
        # matching the standard embedding's init scale (~initializer_range). Output is linear in
        # the final layer for both mlp (ffn[-1]) and tfenc (out_proj), so scaling it scales output.
        with torch.no_grad():
            idx = torch.randint(0, self.vocab, (4096,))
            cur = self.forward(idx).float().std().item()
            if cur > 1e-8:
                k = target_std / cur
                last = self._final()
                last.weight.mul_(k)
                if last.bias is not None:
                    last.bias.mul_(k)

    def forward(self, ids):
        if self._codes.device != ids.device:
            self._codes = self._codes.to(ids.device)
        codes = self._codes[ids]
        if self.enc_type == 'tfenc':
            *lead, n = codes.shape
            x = codes.reshape(-1, self.seq, self.tok_bits)
            x = self.in_proj(x) + self.pos
            x = self.tfenc(x).mean(dim=1)
            return self.out_proj(x).reshape(*lead, self.hidden)
        return self.ffn(codes)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def swap_in_bits(model, nbits, W, seed, enc_type='mlp', ffn_depth=2, out_scale=0.0):
    cfg = model.config
    if getattr(cfg, "tie_word_embeddings", False):
        out = model.get_output_embeddings()
        if out is not None:
            new = nn.Linear(out.weight.shape[1], out.weight.shape[0], bias=False)
            new.weight = nn.Parameter(out.weight.detach().clone())
            model.set_output_embeddings(new); cfg.tie_word_embeddings = False
    dtype = next(model.parameters()).dtype
    bits = BitsEmbedding(cfg.vocab_size, cfg.hidden_size, nbits, W, seed, dtype,
                         enc_type=enc_type, ffn_depth=ffn_depth, out_scale=out_scale)
    dev = model.get_input_embeddings().weight.device
    model.set_input_embeddings(bits.to(dev))
    orig = cfg.vocab_size * cfg.hidden_size
    print(f"[Bits enc={enc_type} depth={ffn_depth} W={W} nbits={nbits} scale={out_scale}] "
          f"table {orig/1e9:.3f}B -> encoder {bits.n_params()/1e6:.2f}M "
          f"= {orig/max(bits.n_params(),1):.0f}x compression", flush=True)
    return bits


def get_dataset(tokenizer, max_len, n_tokens, split_stream):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split=split_stream, streaming=True)
    def gen():
        buf = []
        while True:
            for ex in ds:
                ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
                buf.extend(ids + [tokenizer.eos_token_id])
                while len(buf) >= max_len:
                    yield {"input_ids": buf[:max_len], "labels": buf[:max_len]}
                    buf = buf[max_len:]
    from datasets import IterableDataset
    return IterableDataset.from_generator(gen)
