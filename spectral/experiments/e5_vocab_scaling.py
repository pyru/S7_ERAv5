"""E5 -- "then we can have a vocab of 1M as well without any issues".

Two halves, because the claim has two halves.

  1. ARITHMETIC.  The parameter and training-memory bill of each token
     interface at V5's reference shape and beyond.  Exact, not estimated.

  2. A RUNNING MODEL.  Arithmetic is cheap to write down, so we also build a
     real head-free model with a ONE MILLION token vocabulary and run a
     genuine forward and backward pass through it, on this CPU, and report the
     parameter count and the wall clock.  Nothing about it is a projection.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(8)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from spectral.codecs import SpectralConfig, token_interface_params  # noqa: E402
from spectral.model import ModelConfig, TinyLM, byte_targets, headfree_loss  # noqa: E402
from spectral.data import vocab_byte_list  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "results")

BYTES_PER_PARAM_TRAIN = 16  # bf16 weight + bf16 grad + fp32 master + 2 fp32 moments


def arithmetic():
    rows = []
    for V in (32_000, 131_072, 262_144, 1_000_000):
        for D in (2048, 4096, 8096):
            for scheme, kw in [
                ("dense_untied", {}),
                ("dense_tied", {}),
                ("factorized", {"rank": 512}),
                ("kronecker", {"code_dim": 8192}),
                ("spectral_headfree", {"code_dim": 8192}),
            ]:
                p = token_interface_params(scheme, V, D, **kw)
                tot = p["input"] + p["head"]
                rows.append({
                    "vocab": V, "d_model": D, "scheme": scheme,
                    "input": p["input"], "head": p["head"], "total": tot,
                    "train_gb": round(tot * BYTES_PER_PARAM_TRAIN / 1e9, 3),
                    "vs_dense_untied": round(
                        (V * D * 2) / tot, 2) if tot else None,
                })
    return rows


def million_vocab_demo():
    """Build a head-free model over 1,000,000 tokens and actually run it."""
    real = vocab_byte_list(32000)
    # Extend to 1M with synthetic-but-realistic byte strings.  What matters is
    # that the model must accept a million DISTINCT tokens; where the strings
    # came from is irrelevant to the parameter count, which is the claim.
    import random

    rng = random.Random(0)
    vocab_bytes = list(real)
    while len(vocab_bytes) < 1_000_000:
        a, b = rng.choice(real), rng.choice(real)
        vocab_bytes.append((a + b)[:40])
    vocab_bytes = vocab_bytes[:1_000_000]

    spec = dict(cfg=SpectralConfig(char_dim=64, pos_ch=32, p_grid=32, ladder="dft"))
    out = {}
    for V in (32_000, 131_072, 1_000_000):
        cfg = ModelConfig(vocab_size=V, d_model=256, n_layer=4, n_head=4,
                          d_ff=1024, max_seq=128, input_path="spectral",
                          head="none", codec_kwargs=spec)
        model = TinyLM(cfg, vocab_bytes[:V])
        rep = model.param_report()
        byte_tgt = byte_targets(vocab_bytes[:V], 32)
        x = torch.randint(0, V, (8, 128))
        y = torch.randint(0, V, (8, 128))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        _, h = model(x)  # warm up
        t0 = time.time()
        for _ in range(3):
            _, h = model(x)
            loss = headfree_loss(model, h.reshape(-1, h.shape[-1]), y.reshape(-1), byte_tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / 3

        # what the SAME model would have cost with a dense untied interface
        dense = token_interface_params("dense_untied", V, 256)
        out[str(V)] = {
            "params": rep,
            "on_the_fly_codes": bool(model.on_the_fly),
            "sec_per_step": round(dt, 3),
            "loss_ran": round(float(loss.detach()), 4),
            "dense_untied_token_interface": dense["input"] + dense["head"],
            "reduction_vs_dense": round(
                (dense["input"] + dense["head"]) / rep["token_interface"], 1),
        }
        print(f"  V={V:>9,}  token-interface params={rep['token_interface']:>9,} "
              f"(head={rep['head']})  {dt:.3f}s/step  "
              f"dense would be {dense['input'] + dense['head']:,} "
              f"({out[str(V)]['reduction_vs_dense']}x)", flush=True)
        del model
    return out


def measured_head_cost():
    """Wall-clock of the output path alone, dense head vs spectral decode."""
    d, n = 256, 1024
    h = torch.randn(n, d)
    rows = []
    for V in (8_192, 32_000, 131_072, 262_144):
        W = torch.randn(d, V) * 0.02
        t0 = time.time()
        for _ in range(5):
            lg = h @ W
            lg.logsumexp(-1).sum().backward() if lg.requires_grad else lg.logsumexp(-1).sum()
        dense_t = (time.time() - t0) / 5
        rows.append({"vocab": V, "dense_head_sec": round(dense_t, 4),
                     "dense_head_params": d * V})
    Wc = torch.randn(d, 2048) * 0.02
    t0 = time.time()
    for _ in range(5):
        pred = h @ Wc
        g = pred.reshape(n, 64, 32)
        g = torch.einsum("ncq,gq->ncg", g, torch.randn(32, 32))
        g = torch.einsum("ncg,bc->ngb", g, torch.randn(256, 64))
        g.logsumexp(-1).sum()
    spec_t = (time.time() - t0) / 5
    return {"dense": rows, "spectral_sec": round(spec_t, 4), "spectral_params": 0}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=== arithmetic ===", flush=True)
    rows = arithmetic()
    for r in rows:
        if r["vocab"] == 131_072 and r["d_model"] == 8096:
            print(f"  {r['scheme']:20s} {r['total']:>14,} params  "
                  f"{r['train_gb']:>7.2f} GB", flush=True)
    print("=== 1M vocab, real forward+backward ===", flush=True)
    demo = million_vocab_demo()
    print("=== measured output-path cost ===", flush=True)
    head = measured_head_cost()

    out = {"arithmetic": rows, "million_vocab_demo": demo, "head_cost": head}
    with open(os.path.join(RESULTS, "e5_vocab_scaling.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/e5_vocab_scaling.json")


if __name__ == "__main__":
    main()
