"""Fast self-check of every property the write-up claims about the codecs.

Runs in seconds and needs no corpus, so it is the thing to run first if any of
this is ever picked up again.
"""
from __future__ import annotations

import random
import sys

import torch

sys.path.insert(0, __file__.rsplit("spectral", 1)[0])

from spectral.codecs import (  # noqa: E402
    KroneckerCodec, KroneckerConfig, SpectralCodec, SpectralConfig,
    position_basis, token_interface_params,
)

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def main():
    random.seed(0)
    torch.manual_seed(0)
    words = ["train", "training", "the", "a", "internationalization",
             "अंतर्राष्ट्रीयकरण", "தமிழ்", "ಕನ್ನಡದಲ್ಲಿ", "తెలుగు", "ইতিহাস"]
    seqs = [w.encode() for w in words]

    print("\n[1] the DFT ladder is exactly orthonormal")
    sc = SpectralCodec(SpectralConfig(pos_ch=32, p_grid=32, ladder="dft"))
    check("condition number == 1", abs(sc.pos_cond - 1.0) < 1e-3, f"{sc.pos_cond:.6f}")

    print("\n[2] a geometric ladder is singular -- the trap this design avoids")
    import math

    import numpy as np
    # The RoPE-style ladder that is the obvious first guess, rebuilt here
    # because codecs.py no longer offers it: omega_k = pi * p_max^(-k/(K-1)).
    K = 16
    w = math.pi * np.power(64.0, -np.arange(K) / (K - 1))
    psi = position_basis(32, 32, "dft")
    ang = np.arange(32)[:, None] * w[None, :]
    geo = np.concatenate([np.cos(ang), np.sin(ang)], 1)
    check("sin(pi*p) is a dead channel", abs(np.sin(w[0] * np.arange(32))).max() < 1e-9,
          f"omega_0 = {w[0]:.4f}")
    check("geometric Gram is ill-conditioned",
          np.linalg.cond(geo.T @ geo) > 1e8, f"{np.linalg.cond(geo.T @ geo):.2e}")
    check("DFT Gram is well-conditioned", np.linalg.cond(psi.T @ psi) < 1.01)

    print("\n[3] round-trip is exact in-window, for every script")
    ok = sc.decode_matches(sc.encode_many(seqs, normalize=False), seqs)
    short = [i for i, s in enumerate(seqs) if len(s) < 32]
    check("all tokens under 32 bytes round-trip",
          bool(ok[short].all()), f"{int(ok[short].sum())}/{len(short)}")

    print("\n[4] truncation destroys, folding does not")
    kc = KroneckerCodec(KroneckerConfig(pos_dim=32))
    long = "இருப்பினும்,".encode()          # 34 bytes
    prefix = long[:32]
    kd = (kc.encode_many([long], normalize=False)
          - kc.encode_many([prefix], normalize=False)).abs().max().item()
    sd = (sc.encode_many([long], normalize=False)
          - sc.encode_many([prefix], normalize=False)).abs().max().item()
    check("Kronecker maps a token onto its own 32-byte prefix", kd == 0.0, f"delta {kd:.2e}")
    check("spectral keeps them apart", sd > 1e-3, f"delta {sd:.4f}")

    print("\n[5] a multi-modulus ladder resolves past its channel count")
    v3 = SpectralCodec(SpectralConfig(pos_ch=32, p_grid=96, ladder="vernier3"))
    rnd = [bytes(random.randrange(1, 256) for _ in range(56)) for _ in range(200)]
    kr = kc.decode_matches(kc.encode_many(rnd, normalize=False), rnd).float().mean()
    vr = v3.decode_matches(v3.encode_many(rnd, normalize=False), rnd).float().mean()
    check("Kronecker cannot decode 56 bytes at all", kr.item() == 0.0)
    check("vernier3 decodes 56 bytes from 32 channels", vr.item() > 0.8, f"{vr:.3f}")

    print("\n[6] length is unbounded on the encode side")
    huge = [b"x" * 500]
    check("a 500-byte token encodes without error",
          sc.encode_many(huge, normalize=False).shape == (1, 8192))

    print("\n[7] the token interface does not contain |V|")
    a = token_interface_params("spectral_headfree", 32_000, 256, code_dim=2048)
    b = token_interface_params("spectral_headfree", 1_000_000, 256, code_dim=2048)
    check("params identical at 32k and 1M vocab", a == b, f"{a['input'] + a['head']:,}")
    check("head is zero", a["head"] == 0)

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
