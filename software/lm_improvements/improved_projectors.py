"""
Custom 001 projectors for LM PPL improvements (ideas 3, 1, 2).

Idea 3: BinaryResidualAE
  e = AE3(bits) + Linear(bits)        # residual skip + 3-layer autoencoder

Idea 1: BinarySemHash
  Replace random subword->bits with 64-bit codes obtained by ITQ
  (iterative quantization) of a pretrained Subword embedding table.

Idea 2 is NOT a projector change; it adds an MSE-to-subword-embedding
auxiliary loss during training. See run_improved_lm.py for that.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _one_hot_bits(num_bits: int) -> nn.Embedding:
    oh = F.one_hot(torch.arange(2)).to(torch.float32)
    emb = nn.Embedding(2, 2, _weight=oh)
    emb.requires_grad_(False)
    return emb


class BinaryResidualAE(nn.Module):
    """Idea 3: 3-layer AE with residual bit->embed shortcut.

    Encoder: 2n -> 512 -> 256 -> d
    Decoder: d -> 256 -> 512 -> 2n  (aux reconstruction loss during training)
    Residual: Linear(2n, d) added to encoder output.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 num_bits: int = 64, relu_dropout: float = 0.1,
                 subword_binary_file: str = '',
                 recon_weight: float = 0.05):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout
        self.recon_weight = recon_weight

        if subword_binary_file and os.path.isfile(subword_binary_file):
            mapping = torch.load(subword_binary_file)
            if mapping.shape[0] < num_embeddings:
                extra = torch.randint(0, 2, (num_embeddings - mapping.shape[0], num_bits), dtype=torch.long)
                mapping = torch.cat([mapping, extra], dim=0)
            elif mapping.shape[0] > num_embeddings:
                mapping = mapping[:num_embeddings]
        else:
            mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
            if subword_binary_file:
                os.makedirs(os.path.dirname(subword_binary_file) or '.', exist_ok=True)
                torch.save(mapping, subword_binary_file)
        self.register_buffer('subword2binary', mapping)
        self.binary_embedding = _one_hot_bits(num_bits)

        in_dim = 2 * num_bits
        # Encoder
        self.enc1 = nn.Linear(in_dim, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, embedding_dim)
        # Decoder (aux reconstruction)
        self.dec1 = nn.Linear(embedding_dim, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, in_dim)
        # Residual shortcut
        self.residual = nn.Linear(in_dim, embedding_dim, bias=False)
        self.out_norm = nn.LayerNorm(embedding_dim)

        for m in [self.enc1, self.enc2, self.dec1, self.dec2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_out')
            nn.init.constant_(m.bias, 0.01)
        for m in [self.enc3, self.dec3]:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.residual.weight)

        subword = torch.arange(num_embeddings)
        self.register_buffer('subword', subword)
        self.weight = self.binary_embedding.weight  # vocab_compat attr
        self._last_recon = None

    def _flatten_bits(self, ids: torch.Tensor) -> torch.Tensor:
        bits = self.subword2binary[ids]                   # (..., num_bits)
        emb = self.binary_embedding(bits)                 # (..., num_bits, 2)
        return emb.reshape(*ids.shape, -1)                # (..., 2*num_bits)

    def forward(self, input: torch.Tensor, dropout: float = None) -> torch.Tensor:
        x = self._flatten_bits(input)
        drop = self.relu_dropout if dropout is None else dropout

        h = F.gelu(self.enc1(x)); h = F.dropout(h, p=drop, training=self.training)
        h = F.gelu(self.enc2(h)); h = F.dropout(h, p=drop, training=self.training)
        enc = self.enc3(h)

        out = enc + self.residual(x)
        out = self.out_norm(out)

        if self.training:
            d = F.gelu(self.dec1(enc)); d = F.dropout(d, p=drop, training=self.training)
            d = F.gelu(self.dec2(d));  d = F.dropout(d, p=drop, training=self.training)
            self._last_recon = (self.dec3(d), x.detach())
        else:
            self._last_recon = None
        return out

    def aux_recon_loss(self) -> torch.Tensor:
        if self._last_recon is None:
            return torch.tensor(0.0)
        pred, tgt = self._last_recon
        return F.mse_loss(pred, tgt) * self.recon_weight

    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)


def itq_hash(vectors: torch.Tensor, num_bits: int, n_iter: int = 50) -> torch.Tensor:
    """Iterative Quantization (Gong & Lazebnik 2011): learn a rotation that
    aligns PCA-projected data with the vertices of the {-1,+1}^k hypercube.
    Returns (N, num_bits) binary tensor (0/1)."""
    V = vectors.float()
    V = V - V.mean(dim=0, keepdim=True)
    # PCA to num_bits dims
    U, S, Vt = torch.linalg.svd(V, full_matrices=False)
    pc = Vt[:num_bits].T                                 # (D, num_bits)
    Y = V @ pc                                           # (N, num_bits)
    # Random rotation
    R = torch.linalg.qr(torch.randn(num_bits, num_bits))[0]
    for _ in range(n_iter):
        YR = Y @ R
        B = torch.sign(YR)
        B[B == 0] = 1
        U2, _, V2t = torch.linalg.svd(B.T @ Y, full_matrices=False)
        R = (U2 @ V2t).T
    B = torch.sign(Y @ R)
    B[B == 0] = 1
    return ((B + 1) / 2).long()                          # {0,1}


class BinarySemHash(nn.Module):
    """Idea 1: subword->bits initialized via ITQ of a pretrained Subword
    embedding table. Projector is the same 2-layer FFN as in the paper
    (interdim 4096) so the only change vs baseline is the binary mapping."""

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 num_bits: int = 64, interdim: str = '4096',
                 relu_dropout: float = 0.1,
                 pretrained_emb_file: str = '',
                 subword_binary_file: str = ''):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_bits = num_bits
        self.relu_dropout = relu_dropout

        # --- Build / load binary mapping ---
        if subword_binary_file and os.path.isfile(subword_binary_file):
            mapping = torch.load(subword_binary_file)
            print(f"[BinarySemHash] loaded existing mapping from {subword_binary_file}")
        elif pretrained_emb_file and os.path.isfile(pretrained_emb_file):
            print(f"[BinarySemHash] building ITQ mapping from {pretrained_emb_file}")
            ckpt = torch.load(pretrained_emb_file, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'embedding.weight' in ckpt:
                pt = ckpt['embedding.weight']
            else:
                raise ValueError("pretrained file must be a state_dict with embedding.weight")
            pt = pt[:num_embeddings].float()
            mapping = itq_hash(pt, num_bits)
            if subword_binary_file:
                os.makedirs(os.path.dirname(subword_binary_file) or '.', exist_ok=True)
                torch.save(mapping, subword_binary_file)
                print(f"[BinarySemHash] saved ITQ mapping to {subword_binary_file}")
        else:
            mapping = torch.randint(0, 2, (num_embeddings, num_bits), dtype=torch.long)
        self.register_buffer('subword2binary', mapping)
        self.binary_embedding = _one_hot_bits(num_bits)

        in_dim = 2 * num_bits
        dims = [in_dim] + [int(x) for x in str(interdim).split('_')] + [embedding_dim]
        layers = []
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
        self.linears = nn.ModuleList(layers)
        self.layernum = len(self.linears)
        for i, lin in enumerate(self.linears):
            if i + 1 != self.layernum:
                nn.init.kaiming_normal_(lin.weight, nonlinearity='relu', mode='fan_out')
                nn.init.constant_(lin.bias, 0.01)
            else:
                nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)
        self.out_norm = nn.LayerNorm(embedding_dim)
        subword = torch.arange(num_embeddings)
        self.register_buffer('subword', subword)
        self.weight = self.binary_embedding.weight

    def forward(self, input: torch.Tensor, dropout: float = None) -> torch.Tensor:
        bits = self.subword2binary[input]
        emb = self.binary_embedding(bits)
        x = emb.reshape(*input.shape, -1)
        drop = self.relu_dropout if dropout is None else dropout
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i + 1 != self.layernum:
                x = F.gelu(x)
                x = F.dropout(x, p=drop, training=self.training)
        return self.out_norm(x)

    def aux_recon_loss(self) -> torch.Tensor:
        return torch.tensor(0.0)

    def construct_matrix_for_output_layer(self) -> torch.Tensor:
        return self.forward(self.subword, dropout=0)
