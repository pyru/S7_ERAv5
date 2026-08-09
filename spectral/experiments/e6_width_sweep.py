"""E6 -- the measurement that decides whether the head can actually go.

E4 showed a head-free model works but costs ~0.6 bits/byte against a dense-head
control at d_model = 256.  E3b explained why: a linear squeeze to 256 dimensions
recovers only 62 % of the 8192-dimensional codes, so at that width the model
physically cannot emit the code it needs.  E3b also says the bottleneck is gone
by d_model ~1536.

That turns into a falsifiable prediction, recorded here BEFORE the run:

    PREDICTION.  The head-free penalty shrinks monotonically with d_model, and
    it shrinks in step with the E3b recovery curve -- which reads 0.62 at 256,
    0.89 at 512 and ~0.98 at 768.  If the penalty is instead flat in d_model,
    the rank bottleneck is NOT the operative cause and the head-free design is
    simply worse.

Both arms are retrained at every width so the comparison is like-for-like, and
every arm is scored two ways:

  factorised  sum_p log p(byte_p)         -- an UPPER BOUND on the real cost,
                                             because mass leaks to non-tokens
  exact       renormalised over V         -- directly comparable to the dense
                                             head (see spectral/scoring.py)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

torch.set_num_threads(8)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from spectral.codecs import SpectralConfig  # noqa: E402
from spectral.data import Batcher, build_stream, vocab_byte_list  # noqa: E402
from spectral.model import ModelConfig, TinyLM, byte_targets, headfree_loss  # noqa: E402
from spectral.scoring import VocabScorer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "results")

VOCAB = 32000
STEPS = int(os.environ.get("STEPS", 600))
SEQ, BATCH = 128, 8
# 768 was in the plan and was abandoned: on this 8-core CPU it ran at ~41 s/step
# against ~2 s/step at 512, which is far more than the FLOP ratio and is almost
# certainly memory pressure.  600 steps would have cost about seven hours per
# arm.  The result below therefore spans 256 -> 512 only, which is stated as a
# limitation rather than papered over.
WIDTHS = [int(x) for x in os.environ.get("WIDTHS", "256,512").split(",")]
EVAL_EVERY = 150
EVAL_BATCHES = 6
EXACT_BATCHES = 4          # positions used for the exact renormalised score

SPEC = dict(cfg=SpectralConfig(char_dim=256, pos_ch=32, p_grid=32, ladder="dft"))


def make_cfg(d_model: int, head_free: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, d_model=d_model, n_head=max(4, d_model // 64),
        n_layer=4, d_ff=4 * d_model, max_seq=SEQ,
        input_path="spectral" if head_free else "dense",
        head="none" if head_free else "dense",
        codec_kwargs=SPEC if head_free else {},
    )


@torch.no_grad()
def evaluate(model, batches, byte_tgt, nbytes, scorer=None):
    model.eval()
    nats = nbyte = 0.0
    exact_nats = exact_nbyte = 0.0
    for i, (x, y) in enumerate(batches):
        logits, h = model(x)
        fy = y.reshape(-1)
        if model.cfg.head == "none":
            flat_h = h.reshape(-1, h.shape[-1])
            nats += headfree_loss(model, flat_h, fy, byte_tgt, reduction="sum").item()
            if scorer is not None and i < EXACT_BATCHES:
                exact_nats += scorer.nll(model.char_logits(flat_h), fy)
                exact_nbyte += nbytes[fy].sum().item()
        else:
            nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), fy,
                                    reduction="sum").item()
            if i < EXACT_BATCHES:
                exact_nats += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1])[: fy.numel()], fy,
                    reduction="sum").item()
                exact_nbyte += nbytes[fy].sum().item()
        nbyte += nbytes[fy].sum().item()
    model.train()
    out = {"bits_per_byte": nats / nbyte / math.log(2)}
    if exact_nbyte:
        out["bits_per_byte_exact"] = exact_nats / exact_nbyte / math.log(2)
    return out


def run(d_model, head_free, batcher, vocab_bytes, byte_tgt, nbytes, scorer):
    name = f"d{d_model}_{'headfree' if head_free else 'dense'}"
    torch.manual_seed(0)
    batcher.reset()
    model = TinyLM(make_cfg(d_model, head_free), vocab_bytes)
    rep = model.param_report()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=STEPS,
                                                pct_start=0.1)
    val = batcher.val_batches(EVAL_BATCHES, batch=8)
    curve, t0 = [], time.time()
    for step in range(STEPS):
        x, y = batcher.train_batch()
        logits, h = model(x)
        fy = y.reshape(-1)
        loss = (headfree_loss(model, h.reshape(-1, h.shape[-1]), fy, byte_tgt)
                if head_free else
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), fy))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            ev = evaluate(model, val, byte_tgt, nbytes)
            ev["step"] = step
            curve.append(ev)
            print(f"  [{name}] step {step:4d} bpb={ev['bits_per_byte']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    final = evaluate(model, val, byte_tgt, nbytes, scorer=scorer)
    print(f"  [{name}] FINAL bpb={final['bits_per_byte']:.4f} "
          f"exact={final.get('bits_per_byte_exact', float('nan')):.4f} "
          f"params={rep['total']:,} head={rep['head']:,}", flush=True)
    return {"name": name, "d_model": d_model, "head_free": head_free,
            "params": rep, "final": final, "curve": curve}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    stream = build_stream(VOCAB)
    vocab_bytes = vocab_byte_list(VOCAB)
    batcher = Batcher(stream, SEQ, BATCH)
    byte_tgt = byte_targets(vocab_bytes, 32)
    nbytes = torch.tensor([max(1, len(b)) for b in vocab_bytes], dtype=torch.float32)
    scorer = VocabScorer(byte_tgt)

    path = os.path.join(RESULTS, "e6_width_sweep.json")
    out = {"steps": STEPS, "vocab": VOCAB, "seq": SEQ, "batch": BATCH,
           "prediction": ("head-free penalty shrinks monotonically with d_model, "
                          "tracking the E3b recovery curve (0.62 / 0.89 / 0.98 "
                          "at 256 / 512 / 768)"),
           "e3b_recovery": {"256": 0.6211, "512": 0.8912, "768": 0.9767},
           "runs": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            out["runs"] = json.load(fh).get("runs", [])
    done = {r["name"] for r in out["runs"]}

    for d in WIDTHS:
        for hf in (False, True):
            nm = f"d{d}_{'headfree' if hf else 'dense'}"
            if nm in done:
                print(f"skip {nm} (already in results)", flush=True)
                continue
            print(f"\n=== {nm} ===", flush=True)
            out["runs"].append(
                run(d, hf, batcher, vocab_bytes, byte_tgt, nbytes, scorer))
            out["runs"].sort(key=lambda r: (r["d_model"], r["head_free"]))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2)

    print("\n=== the gap, by width ===", flush=True)
    by = {(r["d_model"], r["head_free"]): r for r in out["runs"]}
    for d in sorted({r["d_model"] for r in out["runs"]}):
        if (d, False) in by and (d, True) in by:
            de, hf = by[(d, False)]["final"], by[(d, True)]["final"]
            g = hf["bits_per_byte"] - de["bits_per_byte"]
            ge = hf.get("bits_per_byte_exact", float("nan")) - \
                de.get("bits_per_byte_exact", float("nan"))
            print(f"  d_model {d:5d}: dense {de['bits_per_byte']:.4f}  "
                  f"headfree {hf['bits_per_byte']:.4f}  gap {g:+.4f}  "
                  f"| exact gap {ge:+.4f}", flush=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
