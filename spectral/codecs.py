"""Byte-level token codecs: the shipped Kronecker grid, and the spectral
(Fourier) alternative proposed in this work.

==============================================================================
THE ONE IDEA
==============================================================================
The shipped Kronecker codec builds a token's code as a sum of outer products
of two *one-hot* vectors:

    kappa(w) = (1/sqrt(L)) * sum_p  onehot256(byte_p)  (x)  onehot32(p)

Both axes are one-hot.  A one-hot basis is maximally sharp and maximally
inflexible: you need one column per position you wish to represent, so the
number of columns *is* the maximum token length.  32 columns, 32 bytes, hard
wall, silent collisions past it.

This module keeps the Kronecker *skeleton* -- a sum over positions of an outer
product -- and replaces the one-hot position basis with a **wave basis**:

    S(w) = (1/sqrt(L)) * sum_p  phi(byte_p)  (x)  psi(p)

    psi(p) = [ .. cos(w_k p), sin(w_k p) .. ]   (a bank of K frequencies)
    phi(b) = a fixed row of an orthogonal character basis

Every character contributes a wave packet, and the word is their
superposition.  That is Problem 4, literally.

==============================================================================
WHY THIS IS NOT COSMETIC:  TRUNCATION vs ALIASING
==============================================================================
For L <= N the two codes carry *identical information* -- psi is an orthogonal
change of basis on the very same grid.  The difference appears only past the
window, and it is total:

  * one-hot position (Kronecker) has no column for p >= 32, so byte 32 and
    everything after it is MULTIPLIED BY ZERO.  Information destroyed.
    Two tokens sharing a 32-byte prefix become bit-identical forever.  Shared
    prefixes are not a rare accident in language -- they are what morphology
    IS.  The failure mode is aligned with the data.

  * a wave basis is periodic, not finite: psi(p) is defined for every p, so
    byte 32 lands on top of byte 0 and ADDS.  Information is FOLDED, never
    destroyed.  Two tokens collide only if their folded sums coincide, which
    is a numerical coincidence rather than a property of their spelling.

Folding strictly dominates truncation: truncation is the special case of a
fold whose out-of-window term was replaced by zero.  That is the claim, and
experiments/e1_collisions.py measures it on a real multilingual vocabulary.

==============================================================================
AND IT INVERTS
==============================================================================
psi is a known linear map, so recovering the byte at each position is a fixed
matched filter -- a constant matrix with ZERO trainable parameters:

    charlogits[p, b]  =  < code , phi(b) (x) psi(p) >

With an orthonormal ladder this is an exact inverse for L <= N.  That is
Problem 5: the model emits a vector, we read the string straight back out of
it, and no |V|-way softmax appears anywhere in the graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch

EOT = 0  # byte 0x00 never occurs in valid UTF-8 text -> a free end-of-token mark


# ==========================================================================
# Character basis  phi(b)
# ==========================================================================
def character_basis(char_dim: int, n_bytes: int = 256) -> np.ndarray:
    """Fixed, untrained map from a byte value to a `char_dim` vector.

    char_dim == 256 -> identity: exactly the one-hot character axis Kronecker
    uses, and lossless.  char_dim < 256 -> a real-DFT frame, which spreads each
    byte over every channel (incoherent rather than sharp).  Used only when we
    deliberately shrink the code below the model width.
    """
    if char_dim >= n_bytes:
        return np.eye(n_bytes, dtype=np.float64)
    return _real_fourier_frame(n_bytes, char_dim)


# ==========================================================================
# Position basis  psi(p)  -- the design surface
# ==========================================================================
def _real_fourier_frame(n_pos: int, n_ch: int) -> np.ndarray:
    """First `n_ch` real-Fourier components sampled at p = 0..n_pos-1.

    Ordered DC, cos1, sin1, cos2, sin2, ...  When n_ch == n_pos this is the
    complete orthonormal real DFT basis: perfectly conditioned, exactly
    invertible, and periodic with period n_pos (so p and p+n_pos alias).
    """
    p = np.arange(n_pos)[:, None].astype(np.float64)
    cols, k = [], 1
    cols.append(np.full((n_pos, 1), 1.0 / math.sqrt(n_pos)))
    while sum(c.shape[1] for c in cols) < n_ch:
        ang = 2.0 * math.pi * k * p / n_pos
        if 2 * k == n_pos:  # Nyquist: cos only, sin is identically zero
            cols.append(np.cos(ang) / math.sqrt(n_pos))
        else:
            cols.append(np.cos(ang) * math.sqrt(2.0 / n_pos))
            cols.append(np.sin(ang) * math.sqrt(2.0 / n_pos))
        k += 1
    return np.concatenate(cols, axis=1)[:, :n_ch]


def _ladder_moduli(kind: str, n_ch: int) -> list[int]:
    """How the `n_ch` positional channels are split across periodicities.

    Every block after the first gives up its DC column (a constant column is
    the same constant column whatever the modulus, so keeping two of them
    duplicates a channel and makes the Gram matrix singular).  A block of
    modulus m therefore costs m channels if it is first and m-1 after that.
    """
    if kind == "dft":
        return [n_ch]
    if kind == "vernier":
        # Two coprime combs.  A collision now needs agreement mod N1 AND mod
        # N2, so the unambiguous range grows to lcm(N1,N2) = N1*N2 >> N1.
        # A vernier caliper / Chinese-remainder argument, spent on bytes.
        for n1 in range((n_ch + 1) // 2, n_ch):
            n2 = n_ch + 1 - n1
            if n2 >= 2 and math.gcd(n1, n2) == 1:
                return [n1, n2]
        return [n_ch]
    if kind == "vernier3":
        for n1 in range(n_ch // 3, n_ch):
            for n2 in range(3, n1 + 1):
                n3 = n_ch + 2 - n1 - n2
                if n3 < 3:
                    continue
                if math.gcd(n1, n2) == math.gcd(n1, n3) == math.gcd(n2, n3) == 1:
                    return [n1, n2, n3]
        return [n_ch]
    raise ValueError(kind)


def position_basis(n_ch: int, p_grid: int, kind: str = "dft") -> np.ndarray:
    """Psi in R^{p_grid x n_ch}; row p is psi(p).  Defined for EVERY p."""
    mods = _ladder_moduli(kind, n_ch)
    blocks = []
    for i, m in enumerate(mods):
        f = _real_fourier_frame(m, m)                    # m x m, orthonormal
        if i > 0:
            f = f[:, 1:]                                 # drop duplicate DC
        idx = np.arange(p_grid) % m                      # <-- the fold
        blocks.append(f[idx] * math.sqrt(m / n_ch))      # keep total energy ~1
    return np.concatenate(blocks, axis=1)[:, :n_ch]


# ==========================================================================
# Configs
# ==========================================================================
@dataclass
class KroneckerConfig:
    char_dim: int = 256
    pos_dim: int = 32           # the 32-byte window

    @property
    def code_dim(self) -> int:
        return self.char_dim * self.pos_dim

    def describe(self) -> dict:
        d = asdict(self)
        d.update(kind="kronecker", code_dim=self.code_dim)
        return d


@dataclass
class SpectralConfig:
    char_dim: int = 256         # C   (256 == lossless, identical to Kronecker's)
    pos_ch: int = 32            # number of positional wave channels
    p_grid: int = 32            # positions the DECODER resolves (not an encode cap)
    ladder: str = "dft"

    @property
    def code_dim(self) -> int:
        return self.char_dim * self.pos_ch

    def describe(self) -> dict:
        d = asdict(self)
        d.update(kind="spectral", code_dim=self.code_dim)
        return d


# ==========================================================================
# Kronecker codec -- the shipped design, as the control arm
# ==========================================================================
class KroneckerCodec:
    name = "kronecker"

    def __init__(self, cfg: KroneckerConfig | None = None):
        self.cfg = cfg or KroneckerConfig()
        self.code_dim = self.cfg.code_dim

    def encode_many(self, byte_seqs, normalize: bool = True) -> torch.Tensor:
        C, P = self.cfg.char_dim, self.cfg.pos_dim
        n = len(byte_seqs)
        tok_idx, byte_val, pos_idx, inv_scale = [], [], [], torch.ones(n)
        for i, bs in enumerate(byte_seqs):
            L = min(len(bs), P)          # <-- the wall.  Bytes past P vanish.
            if L == 0:
                continue
            tok_idx.extend([i] * L)
            byte_val.extend(b % C for b in bs[:L])
            pos_idx.extend(range(L))
            inv_scale[i] = 1.0 / math.sqrt(L)
        out = torch.zeros(n * C * P, dtype=torch.float32)
        if tok_idx:
            flat = (
                torch.tensor(tok_idx, dtype=torch.long) * (C * P)
                + torch.tensor(byte_val, dtype=torch.long) * P
                + torch.tensor(pos_idx, dtype=torch.long)
            )
            out.index_add_(0, flat, torch.ones(len(flat)))
        out = out.reshape(n, C * P) * inv_scale[:, None]
        return _znorm(out) if normalize else out

    def to_char_logits(self, codes: torch.Tensor) -> torch.Tensor:
        C, P = self.cfg.char_dim, self.cfg.pos_dim
        return codes.reshape(*codes.shape[:-1], C, P).transpose(-1, -2)

    def decode_ids(self, codes: torch.Tensor):
        """Kronecker IS invertible -- inside its window.  argmax each column.

        Stating this plainly matters.  Problem 5 is not "Kronecker cannot be
        inverted".  It is that (a) inversion dies past 32 bytes, and (b) the
        model never emits the 8192-d code, it emits a d_model vector, and the
        learned 8192 -> d_model projection has no inverse.  The spectral codec
        answers both; this decoder is here so the comparison is fair.
        """
        C, P = self.cfg.char_dim, self.cfg.pos_dim
        grid = codes.reshape(-1, C, P)
        idx = grid.argmax(dim=1)
        # Where does the token end?  A one-hot grid has no terminator, so the
        # length has to be inferred from which columns carry energy.  Compare
        # each column's peak against the token's own global peak rather than
        # against an absolute epsilon: an absolute threshold reports every
        # column "live" as soon as noise is added, which would lose the
        # baseline the noise experiment for a reason that has nothing to do
        # with Kronecker and everything to do with a bad decode rule.
        peak = grid.max(dim=1).values
        dead = peak <= 0.5 * peak.max(dim=1, keepdim=True).values
        return idx, _first_true(dead, P)

    def decode(self, codes: torch.Tensor, stop_at_eot: bool = False):
        return _to_bytes(*self.decode_ids(codes))

    def decode_matches(self, codes: torch.Tensor, byte_seqs) -> torch.Tensor:
        return _chunked_matches(self, codes, byte_seqs, self.cfg.pos_dim, eot=False)


# ==========================================================================
# Spectral codec -- the proposal
# ==========================================================================
class SpectralCodec:
    name = "spectral"

    def __init__(self, cfg: SpectralConfig | None = None):
        self.cfg = cfg or SpectralConfig()
        self.code_dim = self.cfg.code_dim
        C, Q, G = self.cfg.char_dim, self.cfg.pos_ch, self.cfg.p_grid

        self.phi = torch.tensor(character_basis(C), dtype=torch.float32)   # 256 x C
        psi_np = position_basis(Q, G, self.cfg.ladder)                     # G x Q
        self.psi = torch.tensor(psi_np, dtype=torch.float32)

        gram = psi_np.T @ psi_np
        self.pos_cond = float(np.linalg.cond(gram))
        self.char_cond = float(np.linalg.cond(self.phi.numpy().T @ self.phi.numpy()))

        # Matched filter == exact inverse when the basis is orthonormal.
        self.pos_dec = self.psi.clone()      # G x Q
        self.char_dec = self.phi.clone()     # 256 x C

    # ---------------------------------------------------------------- encode
    def _psi_at(self, positions: torch.Tensor) -> torch.Tensor:
        """psi evaluated at ARBITRARY p, including p >= p_grid.

        This is the point of the whole design: the basis is a periodic function
        of p, not a table of rows, so there is no row to run out of.  Position
        40 folds onto position 40 % N and adds.  It is never dropped.
        """
        return self.psi[positions % self.cfg.p_grid]

    def encode_many(self, byte_seqs, normalize: bool = True, with_eot: bool = True):
        C, Q = self.cfg.char_dim, self.cfg.pos_ch
        n = len(byte_seqs)
        tok_idx, byte_val, pos_idx, inv_scale = [], [], [], torch.ones(n)
        for i, bs in enumerate(byte_seqs):
            seq = list(bs) + ([EOT] if with_eot else [])
            L = len(seq)
            if L == 0:
                continue
            # NOTE: no min(L, window) anywhere.  Any length encodes; positions
            # past the grid fold back onto it and ADD.
            tok_idx.extend([i] * L)
            byte_val.extend(b % 256 for b in seq)
            pos_idx.extend(range(L))
            inv_scale[i] = 1.0 / math.sqrt(L)

        tok_idx = torch.tensor(tok_idx, dtype=torch.long)
        byte_val = torch.tensor(byte_val, dtype=torch.long)
        psi = self._psi_at(torch.tensor(pos_idx, dtype=torch.long))    # M x Q

        if C >= 256:  # phi is the identity -> pure scatter-add, no outer product
            out = torch.zeros(n * 256, Q, dtype=torch.float32)
            out.index_add_(0, tok_idx * 256 + byte_val, psi)
            out = out.reshape(n, 256 * Q)
        else:
            out = torch.zeros(n, C * Q, dtype=torch.float32)
            for s in range(0, len(tok_idx), 65536):
                sl = slice(s, s + 65536)
                blk = torch.einsum("mc,mq->mcq", self.phi[byte_val[sl]], psi[sl])
                out.index_add_(0, tok_idx[sl], blk.reshape(blk.shape[0], -1))
        out = out * inv_scale[:, None]
        return _znorm(out) if normalize else out

    # ---------------------------------------------------------------- decode
    def to_char_logits(self, codes: torch.Tensor) -> torch.Tensor:
        """R^{... x code_dim} -> R^{... x p_grid x 256} character logits.

        A FIXED linear map: zero trainable parameters.  This single tensor
        contraction is the entire replacement for a d_model x |V| output head.
        """
        C, Q = self.cfg.char_dim, self.cfg.pos_ch
        g = codes.reshape(*codes.shape[:-1], C, Q)
        g = torch.einsum("...cq,gq->...cg", g, self.pos_dec)   # ... x C x G
        if C >= 256:  # char_dec is the identity -- a transpose, not a matmul
            return g.transpose(-1, -2)
        return torch.einsum("...cg,bc->...gb", g, self.char_dec)  # ... x G x 256

    def decode_ids(self, codes: torch.Tensor):
        idx = self.to_char_logits(codes).argmax(dim=-1)
        idx = idx.reshape(-1, idx.shape[-1])
        return idx, _first_true(idx == EOT, idx.shape[1])

    def decode(self, codes: torch.Tensor, stop_at_eot: bool = True):
        return _to_bytes(*self.decode_ids(codes))

    def decode_matches(self, codes: torch.Tensor, byte_seqs) -> torch.Tensor:
        return _chunked_matches(self, codes, byte_seqs, self.cfg.p_grid - 1, eot=True)


# ==========================================================================
# Vectorised decode helpers.  The census runs over hundreds of thousands of
# tokens, so "did this decode correctly" has to be a tensor op, not a loop.
# ==========================================================================
def _first_true(mask: torch.Tensor, default: int) -> torch.Tensor:
    """Index of the first True in each row, or `default` if there is none."""
    any_true = mask.any(dim=1)
    first = mask.to(torch.uint8).argmax(dim=1)
    return torch.where(any_true, first, torch.full_like(first, default))


def _to_bytes(idx: torch.Tensor, length: torch.Tensor) -> list[bytes]:
    return [bytes(row[:n]) for row, n in zip(idx.tolist(), length.tolist())]


def _chunked_matches(codec, codes, byte_seqs, max_len: int, eot: bool) -> torch.Tensor:
    """Decoding materialises [chunk, p_grid, 256] logits, which is the largest
    tensor in the whole census.  Size the chunk to that, not to the caller."""
    grid = getattr(codec.cfg, "p_grid", None) or codec.cfg.pos_dim
    chunk = max(256, int(2 ** 24 / (grid * 256)))
    parts = []
    for s in range(0, len(byte_seqs), chunk):
        idx, length = codec.decode_ids(codes[s : s + chunk])
        parts.append(_matches(idx, length, byte_seqs[s : s + chunk], max_len, eot))
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.bool)


def _matches(idx, length, byte_seqs, max_len: int, eot: bool) -> torch.Tensor:
    """True where the decode reproduces the original bytes exactly."""
    n, g = idx.shape
    tgt = torch.full((n, g), -1, dtype=torch.long)
    true_len = torch.tensor([len(b) for b in byte_seqs], dtype=torch.long)
    for i, bs in enumerate(byte_seqs):
        seq = list(bs) + ([EOT] if eot else [])
        if len(seq) <= g:
            tgt[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    representable = true_len <= max_len
    valid = tgt >= 0
    agree = ((idx == tgt) | ~valid).all(dim=1)
    return agree & representable & (length == true_len)


def _znorm(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


# ==========================================================================
# Parameter accounting -- the table the assignment asks us to defend
# ==========================================================================
def token_interface_params(scheme: str, vocab: int, d_model: int, **kw) -> dict:
    if scheme == "dense_untied":
        return {"input": vocab * d_model, "head": d_model * vocab}
    if scheme == "dense_tied":
        return {"input": vocab * d_model, "head": 0}
    if scheme == "factorized":
        r = kw.get("rank", 512)
        return {"input": vocab * r + r * d_model, "head": d_model * vocab}
    if scheme == "kronecker":
        return {"input": kw.get("code_dim", 8192) * d_model, "head": d_model * vocab}
    if scheme == "spectral_headfree":
        # the head is W^T -- the same matrix, already counted on the input side
        return {"input": kw.get("code_dim", 8192) * d_model, "head": 0}
    raise ValueError(scheme)
