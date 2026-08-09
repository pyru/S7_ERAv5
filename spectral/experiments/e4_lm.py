"""E4 -- Does a head-free spectral model actually learn language?

Four arms, identical trunk, identical data, identical seed:

  A dense      : dense embedding + dense untied head        (the control)
  B kronecker  : shipped byte codec  + dense head           (Session 7's design)
  C spectral   : spectral codec      + dense head           (isolates the codec)
  D headfree   : spectral codec      + NO head              (the proposal)

The metric is BITS PER BYTE, not perplexity.  Perplexity is per-token and the
arms do not agree on what a token costs to emit -- D spends probability mass on
byte strings outside the vocabulary, which perplexity would silently reward.
Bits per byte asks the one question all four can answer honestly: how many bits
does this model need to encode this text?

Note this measurement is BIASED AGAINST arm D.  D defines a distribution over
all byte strings, so its total mass is spread over strings the other arms never
have to pay for.  Its bits-per-byte is therefore an upper bound on what a
vocabulary-renormalised version would score.  If D ties the control under that
handicap, the real number is better than the one reported.
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

from spectral.codecs import EOT, KroneckerConfig, SpectralConfig, _first_true  # noqa: E402
from spectral.data import Batcher, build_stream, vocab_byte_list  # noqa: E402
from spectral.model import (  # noqa: E402
    ModelConfig,
    TinyLM,
    byte_targets,
    headfree_loss,
)

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RESULTS = os.path.join(ROOT, "results")

VOCAB = 32000
STEPS = int(os.environ.get("STEPS", 1000))
SEQ, BATCH = 128, 8
EVAL_BATCHES = 6
EVAL_EVERY = 125


def arms(vocab_bytes):
    """Five arms chosen so every adjacent pair isolates exactly one variable.

      A -> B   dense table      vs byte codec        (does a byte code work at all)
      B -> C   Kronecker        vs spectral          (the codec, at EQUAL budget)
      C -> D   dense head       vs no head           (the head, at EQUAL codec)
      D -> E   8192-d code      vs 2048-d code       (how small can the code go)

    Kronecker keeps its shipped char_dim=256.  Folding byte values into 64
    channels, as an earlier draft of this file did, makes 0xE0 collide with
    the space character and would have handicapped the baseline for no reason.
    """
    kron = dict(cfg=KroneckerConfig(char_dim=256, pos_dim=32))          # 8192
    spec8k = dict(cfg=SpectralConfig(char_dim=256, pos_ch=32, p_grid=32, ladder="dft"))
    spec2k = dict(cfg=SpectralConfig(char_dim=64, pos_ch=32, p_grid=32, ladder="dft"))
    base = dict(vocab_size=VOCAB, d_model=256, n_layer=4, n_head=4, d_ff=1024, max_seq=SEQ)
    return {
        "A_dense": ModelConfig(**base, input_path="dense", head="dense"),
        "B_kron8192_head": ModelConfig(**base, input_path="kronecker", head="dense",
                                       codec_kwargs=kron),
        "C_spec8192_head": ModelConfig(**base, input_path="spectral", head="dense",
                                       codec_kwargs=spec8k),
        "D_spec8192_headfree": ModelConfig(**base, input_path="spectral", head="none",
                                           codec_kwargs=spec8k),
        "E_spec2048_headfree": ModelConfig(**base, input_path="spectral", head="none",
                                           codec_kwargs=spec2k),
    }


def token_nbytes(vocab_bytes) -> torch.Tensor:
    """Denominator for bits-per-byte: real UTF-8 bytes each token stands for."""
    return torch.tensor([max(1, len(b)) for b in vocab_bytes], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, batches, byte_tgt, nbytes):
    model.eval()
    nats, nbyte, ntok, ncorrect = 0.0, 0.0, 0, 0
    for x, y in batches:
        logits, h = model(x)
        flat_y = y.reshape(-1)
        if model.cfg.head == "none":
            nats += headfree_loss(model, h.reshape(-1, h.shape[-1]), flat_y,
                                  byte_tgt, reduction="sum").item()
            pred = model.char_logits(h.reshape(-1, h.shape[-1])).argmax(-1)
            tgt = byte_tgt[flat_y]
            mask = tgt != -100
            # Positions past the token's end are ignored, not required to
            # match.  Requiring them (`& mask`) makes this identically zero for
            # every token shorter than the grid, which is every token.
            ncorrect += ((pred == tgt) | ~mask).all(dim=-1).sum().item()
        else:
            nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), flat_y,
                                    reduction="sum").item()
            ncorrect += (logits.reshape(-1, logits.shape[-1]).argmax(-1) == flat_y).sum().item()
        nbyte += nbytes[flat_y].sum().item()
        ntok += flat_y.numel()
    model.train()
    return {
        "nats_per_token": nats / ntok,
        "bits_per_byte": nats / nbyte / math.log(2),
        "token_acc": ncorrect / ntok,
    }


@torch.no_grad()
def sample_decodes(model, batch, vocab_bytes, vocab_set, n=24):
    """What the head-free model literally emits: a STRING, read straight out
    of the predicted vector by the fixed inverse transform.  No vocabulary is
    consulted at any point, so nothing constrains the output to be a real
    token -- which is exactly why this can generate out-of-vocabulary text."""
    x, y = batch
    _, h = model(x)
    flat_h = h.reshape(-1, h.shape[-1])[:n]
    # char_logits already IS the decoded grid -- feeding it back through
    # decode_ids would try to read it as a raw code and reshape wrongly.
    idx = model.char_logits(flat_h).argmax(-1)
    length = _first_true(idx == EOT, idx.shape[1])
    out = []
    for i, (row, ln) in enumerate(zip(idx.tolist(), length.tolist())):
        pred = bytes(row[:ln])
        gold = vocab_bytes[int(y.reshape(-1)[i])]
        out.append({
            "predicted": pred.decode("utf-8", errors="replace"),
            "gold": gold.decode("utf-8", errors="replace"),
            "exact": pred == gold,
            "in_vocab": pred in vocab_set,
        })
    return out


def run_arm(name, cfg, batcher, vocab_bytes, byte_tgt, nbytes, steps):
    torch.manual_seed(0)
    batcher.reset()          # every arm sees the same batches in the same order
    model = TinyLM(cfg, vocab_bytes)
    rep = model.param_report()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    val_batches = batcher.val_batches(EVAL_BATCHES, batch=8)
    curve, t0 = [], time.time()

    for step in range(steps):
        x, y = batcher.train_batch()
        logits, h = model(x)
        flat_y = y.reshape(-1)
        if cfg.head == "none":
            loss = headfree_loss(model, h.reshape(-1, h.shape[-1]), flat_y,
                                 byte_tgt, reduction="mean")
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), flat_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % EVAL_EVERY == 0 or step == steps - 1:
            ev = evaluate(model, val_batches, byte_tgt, nbytes)
            ev["step"] = step
            ev["train_loss"] = loss.item()
            ev["elapsed"] = round(time.time() - t0, 1)
            curve.append(ev)
            print(f"  [{name}] step {step:5d} bpb={ev['bits_per_byte']:.4f} "
                  f"acc={ev['token_acc']:.4f} ({ev['elapsed']:.0f}s)", flush=True)

    final = evaluate(model, val_batches, byte_tgt, nbytes)
    if cfg.head == "none":
        final["samples"] = sample_decodes(model, val_batches[0], vocab_bytes,
                                          set(vocab_bytes))
    return {"arm": name, "params": rep, "final": final, "curve": curve,
            "config": {"input_path": cfg.input_path, "head": cfg.head,
                       "vocab": cfg.vocab_size, "d_model": cfg.d_model}}, model


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("building stream...", flush=True)
    stream = build_stream(VOCAB)
    vocab_bytes = vocab_byte_list(VOCAB)
    print(f"stream {len(stream):,} tokens, vocab {len(vocab_bytes)}", flush=True)
    batcher = Batcher(stream, SEQ, BATCH)
    byte_tgt = byte_targets(vocab_bytes, 32)
    nbytes = token_nbytes(vocab_bytes)

    path = os.path.join(RESULTS, "e4_lm.json")
    out = {"steps": STEPS, "vocab": VOCAB, "seq": SEQ, "batch": BATCH, "arms": []}
    only = set(os.environ.get("ARMS", "").split(",")) - {""}
    if only and os.path.exists(path):  # re-running a subset: keep the others
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
        out["arms"] = [a for a in prev.get("arms", []) if a["arm"] not in only]

    for name, cfg in arms(vocab_bytes).items():
        if only and name not in only:
            continue
        print(f"\n=== {name} ===", flush=True)
        res, _ = run_arm(name, cfg, batcher, vocab_bytes, byte_tgt, nbytes, STEPS)
        p = res["params"]
        print(f"  params: input={p['input']:,} head={p['head']:,} "
              f"trunk={p['trunk']:,} total={p['total']:,}", flush=True)
        out["arms"].append(res)
        out["arms"].sort(key=lambda a: a["arm"])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    print("\nwrote results/e4_lm.json")


if __name__ == "__main__":
    main()
