"""A small transformer with a pluggable token interface.

Three input paths and three output paths can be mixed, so the embedding and
the head can be attributed separately:

  input : dense | kronecker | spectral
  head  : dense | tied | none        ("none" = the fixed spectral decode)

Everything between the two doors is identical across arms -- same depth, same
width, same RoPE, same init, same data, same seed -- so any difference in the
result is attributable to the token interface and nothing else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codecs import EOT, KroneckerCodec, SpectralCodec


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 1024
    max_seq: int = 128
    input_path: str = "dense"        # dense | kronecker | spectral
    head: str = "dense"              # dense | tied | none
    codec_kind: str = "spectral"
    codec_kwargs: dict = field(default_factory=dict)
    untie_head: bool = False         # head="none": give the decode its own W_out


# --------------------------------------------------------------------------
# RoPE for sequence position.  No parameters, and the same frequency-ladder
# idea the codec uses for character position -- one wave basis, two axes.
# --------------------------------------------------------------------------
def rope_tables(seq: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq).float()[:, None] * inv[None, :]
    return torch.cos(t), torch.sin(t)


def apply_rope(x, cos, sin):
    # x: [B, H, T, Dh]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h = cfg.n_head
        self.dh = cfg.d_model // cfg.n_head
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_bytes: list[bytes] | None = None):
        super().__init__()
        self.cfg = cfg
        V, D = cfg.vocab_size, cfg.d_model

        # ---------------- input path -------------------------------------
        self.codec = None
        if cfg.input_path == "dense":
            self.tok_emb = nn.Embedding(V, D)
            nn.init.normal_(self.tok_emb.weight, std=0.02)
        else:
            assert vocab_bytes is not None, "byte codecs need the token strings"
            self.codec = (
                SpectralCodec(**cfg.codec_kwargs)
                if cfg.input_path == "spectral"
                else KroneckerCodec(**cfg.codec_kwargs)
            )
            # A table is a convenience, not a requirement.  The code is a pure
            # function of the token's bytes, so at large |V| we compute it on
            # demand for the few thousand distinct tokens in the batch and
            # never materialise |V| x code_dim at all.  This is what makes a
            # 1M-token vocabulary cost the same as a 32k one.
            self.vocab_bytes = vocab_bytes
            self.on_the_fly = len(vocab_bytes) * self.codec.code_dim > 2 ** 28
            if self.on_the_fly:
                self.register_buffer("code_table", torch.zeros(0), persistent=False)
            else:
                self.register_buffer(
                    "code_table", _build_table(self.codec, vocab_bytes), persistent=False
                )
            # THE only trainable tensor in the input path, and its size does
            # not contain V anywhere.
            self.proj = nn.Linear(self.codec.code_dim, D, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)

        # ---------------- output path ------------------------------------
        self.ln_f = nn.LayerNorm(D)
        if cfg.head == "dense":
            self.lm_head = nn.Linear(D, V, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)
        elif cfg.head == "tied":
            assert cfg.input_path == "dense"
            self.lm_head = nn.Linear(D, V, bias=False)
            self.lm_head.weight = self.tok_emb.weight
        elif cfg.head == "none":
            # No head tensor exists.  Prediction is proj^T followed by the
            # codec's fixed inverse.  One scalar temperature, and that is all.
            assert cfg.input_path == "spectral"
            self.logit_scale = nn.Parameter(torch.tensor(2.0))
            if cfg.untie_head:
                self.out_proj = nn.Linear(D, self.codec.code_dim, bias=False)
                nn.init.normal_(self.out_proj.weight, std=0.02)
            self.register_buffer("pos_dec", self.codec.pos_dec, persistent=False)
            self.register_buffer("char_dec", self.codec.char_dec, persistent=False)
        else:
            raise ValueError(cfg.head)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        cos, sin = rope_tables(cfg.max_seq, D // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # ------------------------------------------------------------------
    def embed(self, ids):
        if self.cfg.input_path == "dense":
            return self.tok_emb(ids)
        if not self.on_the_fly:
            return self.proj(self.code_table[ids])
        uniq, inv = torch.unique(ids.reshape(-1), return_inverse=True)
        codes = self.codec.encode_many(
            [self.vocab_bytes[i] for i in uniq.tolist()], normalize=True
        )
        return self.proj(codes[inv].reshape(*ids.shape, -1))

    def trunk(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.ln_f(x)

    def forward(self, ids):
        h = self.trunk(ids)
        if self.cfg.head == "none":
            return None, h
        return self.lm_head(h), h

    # ------------------------------------------------------------------
    def char_logits(self, h):
        """h -> [..., p_grid, 256] byte logits, using ZERO head parameters.

        pred_code = h @ W   (W is the input projection, reused transposed)
        logits    = fixed inverse transform of pred_code
        """
        cfg = self.codec.cfg
        pred = self.out_proj(h) if self.cfg.untie_head else h @ self.proj.weight
        g = pred.reshape(*pred.shape[:-1], cfg.char_dim, cfg.pos_ch)
        g = torch.einsum("...cq,gq->...cg", g, self.pos_dec)
        g = (g.transpose(-1, -2) if cfg.char_dim >= 256
             else torch.einsum("...cg,bc->...gb", g, self.char_dec))
        return g * self.logit_scale

    # ------------------------------------------------------------------
    def param_report(self) -> dict:
        def n(mod):
            return sum(p.numel() for p in mod.parameters())

        trunk = sum(n(b) for b in self.blocks) + n(self.ln_f)
        if self.cfg.input_path == "dense":
            inp = self.tok_emb.weight.numel()
        else:
            inp = self.proj.weight.numel()
        if self.cfg.head == "dense":
            head = self.lm_head.weight.numel()
        elif self.cfg.head == "tied":
            head = 0
        elif self.cfg.untie_head:
            head = self.out_proj.weight.numel() + 1
        else:
            head = 1  # the temperature scalar, and nothing else
        return {
            "input": inp,
            "head": head,
            "trunk": trunk,
            "total": inp + head + trunk,
            "token_interface": inp + head,
            "token_interface_pct": round(100 * (inp + head) / (inp + head + trunk), 1),
        }


def _build_table(codec, vocab_bytes: list[bytes]) -> torch.Tensor:
    rows = []
    for s in range(0, len(vocab_bytes), 4096):
        rows.append(codec.encode_many(vocab_bytes[s : s + 4096], normalize=True))
    return torch.cat(rows, 0)


def byte_targets(vocab_bytes: list[bytes], p_grid: int) -> torch.Tensor:
    """[V, p_grid] target bytes; -100 past the token's end (ignored by CE).

    Position L holds EOT, which is how the decoder learns where to stop and
    what makes open-vocabulary generation possible at all.
    """
    t = torch.full((len(vocab_bytes), p_grid), -100, dtype=torch.long)
    for i, bs in enumerate(vocab_bytes):
        seq = list(bs)[: p_grid - 1] + [EOT]
        t[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return t


def headfree_loss(model, h, targets_ids, byte_tgt, reduction="sum"):
    """Cross-entropy over the target token's BYTES -- no |V| axis anywhere."""
    logits = model.char_logits(h)                      # [N, G, 256]
    tgt = byte_tgt[targets_ids]                        # [N, G]
    return F.cross_entropy(
        logits.reshape(-1, 256), tgt.reshape(-1), ignore_index=-100,
        reduction=reduction,
    )


def count_flops_head(cfg: ModelConfig, code_dim: int) -> dict:
    """Forward FLOPs of the output path per token -- the other half of the bill."""
    return {
        "dense_head": 2 * cfg.d_model * cfg.vocab_size,
        "spectral_head": 2 * cfg.d_model * code_dim,
    }
