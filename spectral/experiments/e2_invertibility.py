"""E2/E3 -- Is the code actually invertible, and does the inverse survive
contact with a real model?

Problem 5 asks for a codec that runs backwards, so the |V|-way output head can
be deleted.  Three things have to be true for that to work, and each is a
separate measurement here:

  E2  CLEAN INVERSION.  Encode a real token, decode it, get the same bytes.
      Measured against byte length, because that is where Kronecker's window
      turns into a cliff.

  E3a NOISE.  A model does not emit the exact code, it emits an approximation.
      How much error can the decoder absorb before the token changes?

  E3b THE RANK BOTTLENECK -- the one that actually decides whether the idea
      works.  The model emits d_model numbers, not code_dim numbers, so the
      reachable codes lie in a d_model-dimensional subspace.  We compress the
      whole vocabulary's codes to d dimensions (by PCA, which is the best any
      linear projection can do), reconstruct, and decode.  The resulting curve
      says exactly how wide a model has to be before a head-free decoder is
      lossless -- and therefore whether this is a real design or a toy.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from spectral.codecs import (  # noqa: E402
    KroneckerCodec,
    KroneckerConfig,
    SpectralCodec,
    SpectralConfig,
)
from spectral.vocab import load_or_train, vocab_entries, word_entries  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "results")
torch.manual_seed(0)


def codecs():
    return {
        "kron_p32": KroneckerCodec(KroneckerConfig(pos_dim=32)),
        "kron_p64": KroneckerCodec(KroneckerConfig(pos_dim=64)),
        "spec_dft_q32": SpectralCodec(SpectralConfig(pos_ch=32, p_grid=32, ladder="dft")),
        # the 2048-d code the language-model experiment actually trains on
        "spec_dft_2048": SpectralCodec(
            SpectralConfig(char_dim=64, pos_ch=32, p_grid=32, ladder="dft")),
        "spec_vern3_q32": SpectralCodec(
            SpectralConfig(pos_ch=32, p_grid=96, ladder="vernier3")),
        "spec_vern3_q64": SpectralCodec(
            SpectralConfig(pos_ch=64, p_grid=96, ladder="vernier3")),
    }


def _chunks(seqs, codec, n=4096):
    for s in range(0, len(seqs), n):
        blk = seqs[s : s + n]
        yield blk, codec.encode_many(blk, normalize=False)


# ---------------------------------------------------------------- E2
def e2_length_curve(entries):
    """Exact round-trip rate bucketed by true byte length, on real words."""
    buckets = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 56), (57, 80), (81, 999)]

    def bucket(n):
        for i, (lo, hi) in enumerate(buckets):
            if lo <= n <= hi:
                return i
        return len(buckets) - 1

    out = {"buckets": [f"{lo}-{hi if hi < 999 else '80+'}" for lo, hi in buckets],
           "codecs": {}}
    seqs = [e["bytes"] for e in entries]
    tot = defaultdict(int)
    for e in entries:
        tot[bucket(e["nbytes"])] += 1
    out["n_per_bucket"] = [tot[i] for i in range(len(buckets))]

    for name, codec in codecs().items():
        ok = defaultdict(int)
        for blk, codes in _chunks(seqs, codec):
            for hit, b in zip(codec.decode_matches(codes, blk).tolist(), blk):
                if hit:
                    ok[bucket(len(b))] += 1
        out["codecs"][name] = {
            "code_dim": codec.code_dim,
            "rate": [round(ok[i] / tot[i], 4) if tot[i] else None
                     for i in range(len(buckets))],
        }
        print(f"  E2 {name:16s} {out['codecs'][name]['rate']}", flush=True)
    return out


# ---------------------------------------------------------------- E3a
def e3_noise(entries, sigmas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    """Additive Gaussian error on the code, scaled to the code's own RMS."""
    seqs = [e["bytes"] for e in entries[:20000]]
    out = {"sigmas": list(sigmas), "codecs": {}}
    for name, codec in codecs().items():
        rates = []
        for sg in sigmas:
            ok = n = 0
            for blk, codes in _chunks(seqs, codec):
                rms = codes.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                noisy = codes + torch.randn_like(codes) * rms * sg
                ok += int(codec.decode_matches(noisy, blk).sum())
                n += len(blk)
            rates.append(round(ok / n, 4))
        out["codecs"][name] = {"code_dim": codec.code_dim, "rate": rates}
        print(f"  E3a {name:16s} {rates}", flush=True)
    return out


# ---------------------------------------------------------------- E3b
def e3_rank(entries, dims=(32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096)):
    """How wide must d_model be for a head-free decoder to be lossless?

    PCA is the optimal linear map into d dimensions, so this is the best case
    for ANY architecture that squeezes the code through a d_model bottleneck.
    A model narrower than the knee in this curve cannot decode reliably no
    matter how it is trained; a model wider than it loses nothing.
    """
    seqs = [e["bytes"] for e in entries[:20000]]
    out = {"dims": list(dims), "codecs": {}}
    for name, codec in codecs().items():
        # The eigendecomposition is O(code_dim^3).  At 16384 that is ~4 TFLOP
        # in float64 and dominates the whole experiment for no extra insight,
        # since the 8192-dim codecs already bracket the answer.
        if codec.code_dim > 8192:
            out["codecs"][name] = {"code_dim": codec.code_dim, "skipped": True}
            print(f"  E3b {name:16s} skipped (code_dim {codec.code_dim} > 8192)", flush=True)
            continue
        codes = torch.cat([c for _, c in _chunks(seqs, codec)], 0)
        mu = codes.mean(0, keepdim=True)
        X = codes - mu
        cov = ((X.T @ X) / X.shape[0]).double()
        # Kronecker codes are extremely sparse, so the covariance is
        # near-degenerate with a large repeated zero eigenvalue and plain eigh
        # refuses to converge on it.  A ridge on the diagonal is enough to
        # separate the repeats; it shifts every eigenvalue by the same constant
        # and leaves the eigenVECTORS -- the only thing used below -- untouched.
        cov += torch.eye(cov.shape[0], dtype=cov.dtype) * cov.diagonal().mean() * 1e-6
        try:
            evals, evecs = torch.linalg.eigh(cov)
        except Exception as exc:
            print(f"  E3b {name:16s} eigh failed ({exc}); falling back to SVD", flush=True)
            _u, s, vh = torch.linalg.svd(X.double(), full_matrices=False)
            evals, evecs = (s ** 2) / X.shape[0], vh.T
        evals, evecs = evals.float(), evecs.float()
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]
        energy = (evals.clamp_min(0) / evals.clamp_min(0).sum()).cumsum(0)
        rates, retained = [], []
        for d in dims:
            if d > codes.shape[1]:
                rates.append(None); retained.append(None); continue
            P = evecs[:, :d]
            recon = (X @ P) @ P.T + mu
            ok = int(codec.decode_matches(recon, seqs).sum())
            rates.append(round(ok / len(seqs), 4))
            retained.append(round(float(energy[d - 1]), 5))
        out["codecs"][name] = {"code_dim": codec.code_dim, "rate": rates,
                               "energy_retained": retained}
        print(f"  E3b {name:16s} {rates}", flush=True)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "e2_invertibility.json")
    out = {}
    if os.path.exists(path):  # keep whatever already succeeded
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)

    want = set(sys.argv[1:]) or {"e2", "e3a", "e3b"}
    need_words = "e2" in want and "e2_length_words" not in out
    need_bpe = ({"e3a", "e3b"} & want) and not {
        "e3a_noise_bpe32k", "e3b_rank_bpe32k"} <= set(out)

    if need_words:
        out["e2_length_words"] = e2_length_curve(word_entries(120_000))
        _save(path, out)
    if need_bpe:
        bpe = vocab_entries(load_or_train(32000))
        if "e3a" in want and "e3a_noise_bpe32k" not in out:
            out["e3a_noise_bpe32k"] = e3_noise(bpe)
            _save(path, out)
        if "e3b" in want and "e3b_rank_bpe32k" not in out:
            out["e3b_rank_bpe32k"] = e3_rank(bpe)
            _save(path, out)
    _save(path, out)
    print(f"wrote {path} with sections {sorted(out)}")


def _save(path, out):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
