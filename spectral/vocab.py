"""Train a real multilingual BPE tokenizer and expose its vocabulary as bytes.

The collision argument is only worth anything if it is measured on a real
tokenizer's real vocabulary.  Synthetic strings would let us prove whatever we
liked.  So we train byte-level BPE on the multilingual Wikipedia corpus and
census the vocabulary that actually comes out.
"""
from __future__ import annotations

import json
import os
import unicodedata

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")

# Unicode block -> script name, for the per-script collision census.
SCRIPT_RANGES = [
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0000, 0x024F, "Latin"),
]


def script_of(text: str) -> str:
    """Dominant script of a token string."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if unicodedata.category(ch).startswith(("Z", "C")) or ch in " ▁Ġ":
            continue
        name = "Other"
        for lo, hi, nm in SCRIPT_RANGES:
            if lo <= cp <= hi:
                name = nm
                break
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Whitespace"
    return max(counts, key=counts.get)


def corpus_files() -> list[str]:
    return sorted(
        os.path.join(DATA, f) for f in os.listdir(DATA) if f.endswith(".txt")
    )


def train_bpe(vocab_size: int, out_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    # ByteLevel pre-tokenizer: every token maps back to a definite byte string,
    # which is exactly what a byte-level codec needs.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    )
    tok.train(corpus_files(), trainer)
    tok.save(out_path)
    return tok


def load_or_train(vocab_size: int) -> Tokenizer:
    path = os.path.join(DATA, f"bpe_{vocab_size}.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    return train_bpe(vocab_size, path)


# ---------------------------------------------------------------------------
# ByteLevel uses a printable-char surrogate alphabet; invert it to real bytes.
# ---------------------------------------------------------------------------
def _byte_decoder() -> dict[str, int]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _byte_decoder()


def token_bytes(surface: str) -> bytes:
    """Surrogate-alphabet token string -> the real UTF-8 bytes it stands for."""
    try:
        return bytes(_BD[c] for c in surface)
    except KeyError:  # special tokens like <|endoftext|>
        return surface.encode("utf-8")


def vocab_entries(tok: Tokenizer) -> list[dict]:
    """[{id, surface, text, bytes, nbytes, script}] for the whole vocabulary."""
    vocab = tok.get_vocab()
    out = []
    for surface, idx in vocab.items():
        raw = token_bytes(surface)
        text = raw.decode("utf-8", errors="replace")
        out.append(
            {
                "id": idx,
                "surface": surface,
                "text": text,
                "bytes": raw,
                "nbytes": len(raw),
                "script": script_of(text),
            }
        )
    out.sort(key=lambda e: e["id"])
    return out


def word_entries(max_words: int = 250_000) -> list[dict]:
    """The corpus's distinct whitespace-delimited words, as a 'vocabulary'.

    This matters because it is the regime a byte codec actually exists to
    serve.  BPE tokens are short by construction -- BPE stops merging when
    frequency runs out -- so the 32-byte window rarely binds on them.  Words
    are what a byte codec must handle when it meets an unseen string, and what
    an open-vocabulary decoder must be able to emit.  Unlike a BPE vocabulary,
    this one is a property of the language, not of a training run.
    """
    import collections
    import re

    rx = re.compile(r"\S+")
    counts: collections.Counter = collections.Counter()
    for path in corpus_files():
        with open(path, encoding="utf-8") as fh:
            counts.update(rx.findall(fh.read()))
    out = []
    for i, (w, c) in enumerate(counts.most_common(max_words)):
        raw = w.encode("utf-8")
        out.append(
            {
                "id": i,
                "surface": w,
                "text": w,
                "bytes": raw,
                "nbytes": len(raw),
                "count": c,
                "script": script_of(w),
            }
        )
    return out


if __name__ == "__main__":
    import sys

    vs = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    tok = load_or_train(vs)
    ents = vocab_entries(tok)
    by_script: dict[str, int] = {}
    for e in ents:
        by_script[e["script"]] = by_script.get(e["script"], 0) + 1
    print(json.dumps({"vocab": len(ents), "by_script": by_script}, indent=2))
    longest = sorted(ents, key=lambda e: -e["nbytes"])[:10]
    for e in longest:
        print(e["nbytes"], e["script"], repr(e["text"]))
