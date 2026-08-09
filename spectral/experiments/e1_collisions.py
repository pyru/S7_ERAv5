"""E1 -- Collision census on a REAL multilingual vocabulary.

Session 7 section 8 asks for exactly one number: how many tokens of the real
vocabulary become indistinguishable under the shipped 32-byte window, per
script.  This produces that number, and produces it for the spectral codec at
the same parameter budget so the two can be compared rather than admired.

A "collision" here is the strong kind: two different tokens whose codes are
bit-identical, so the projection sees one input for two tokens and no amount
of training can ever separate them.

We also report round-trip decode accuracy, which is the finer-grained version
of the same question -- it measures how much of the token survives the codec,
rather than only whether two tokens happened to land on top of each other.
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
CHUNK = 4096


def build_configs():
    """Each entry: (name, codec, code_dim).  Budget is stated so that no
    comparison is made between designs of different size by accident."""
    return [
        ("kron_p32", KroneckerCodec(KroneckerConfig(pos_dim=32))),
        ("kron_p48", KroneckerCodec(KroneckerConfig(pos_dim=48))),
        ("kron_p64", KroneckerCodec(KroneckerConfig(pos_dim=64))),
        ("spec_dft_q32", SpectralCodec(SpectralConfig(pos_ch=32, p_grid=32, ladder="dft"))),
        ("spec_vern3_q32", SpectralCodec(SpectralConfig(pos_ch=32, p_grid=96, ladder="vernier3"))),
        ("spec_vern3_q64", SpectralCodec(SpectralConfig(pos_ch=64, p_grid=96, ladder="vernier3"))),
    ]


def census(entries, codec, name):
    """Stream the vocabulary through the codec, hashing codes as we go."""
    seen: dict[bytes, int] = {}
    collide_groups: dict[bytes, list[int]] = defaultdict(list)
    decode_ok = defaultdict(int)
    decode_tot = defaultdict(int)

    for s in range(0, len(entries), CHUNK):
        blk = entries[s : s + CHUNK]
        seqs = [e["bytes"] for e in blk]
        codes = codec.encode_many(seqs, normalize=False)
        # quantise before hashing: we are asking "are these the same vector",
        # and float dust from a different summation order is not a difference.
        keys = torch.round(codes * 1e5).to(torch.int32).numpy()
        ok = codec.decode_matches(codes, seqs).tolist()
        for i, e in enumerate(blk):
            k = keys[i].tobytes()
            if k in seen:
                if not collide_groups[k]:
                    collide_groups[k].append(seen[k])
                collide_groups[k].append(e["id"])
            else:
                seen[k] = e["id"]
            decode_tot[e["script"]] += 1
            decode_ok[e["script"]] += int(ok[i])

    by_id = {e["id"]: e for e in entries}
    per_script_collided = defaultdict(int)
    examples = []
    n_collided = 0
    for k, ids in collide_groups.items():
        n_collided += len(ids)
        for i in ids:
            per_script_collided[by_id[i]["script"]] += 1
        if len(examples) < 25:
            examples.append([by_id[i]["text"] for i in ids])

    return {
        "codec": name,
        "code_dim": codec.code_dim,
        "distinct_codes": len(seen),
        "n_tokens": len(entries),
        "n_collided_tokens": n_collided,
        "n_collision_groups": len(collide_groups),
        "collided_pct": 100.0 * n_collided / len(entries),
        "collided_per_script": dict(per_script_collided),
        "decode_exact_per_script": {
            k: round(decode_ok[k] / decode_tot[k], 4) for k in decode_tot
        },
        "decode_exact_overall": round(
            sum(decode_ok.values()) / max(1, sum(decode_tot.values())), 4
        ),
        "examples": examples,
    }


def main(vocab_sizes):
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    targets = [(f"bpe_{V}", V) for V in vocab_sizes] + [("words", None)]
    for label, V in targets:
        print(f"\n=== {label} ===", flush=True)
        if V is None:
            entries = word_entries()
        else:
            entries = vocab_entries(load_or_train(V))
        lens = defaultdict(list)
        for e in entries:
            lens[e["script"]].append(e["nbytes"])
        out[label] = {
            "vocab_size": len(entries),
            "byte_len_stats": {
                k: {
                    "n": len(v),
                    "mean": round(sum(v) / len(v), 2),
                    "max": max(v),
                    "pct_over_32": round(100 * sum(x > 32 for x in v) / len(v), 2),
                    "pct_over_48": round(100 * sum(x > 48 for x in v) / len(v), 2),
                }
                for k, v in sorted(lens.items())
            },
            "codecs": [],
        }
        for name, codec in build_configs():
            r = census(entries, codec, name)
            out[label]["codecs"].append(r)
            print(
                f"  {name:16s} dim={r['code_dim']:6d} "
                f"collided={r['n_collided_tokens']:6d} ({r['collided_pct']:5.2f}%) "
                f"decode={r['decode_exact_overall']:.4f}",
                flush=True,
            )
        with open(os.path.join(RESULTS, "e1_collisions.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nwrote results/e1_collisions.json")


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [8192, 32000, 131072]
    main(sizes)
