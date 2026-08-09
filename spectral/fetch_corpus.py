"""Fetch a real multilingual corpus from Wikipedia.

We need REAL Indic text, not synthetic strings: the whole 32-byte-window
argument turns on real conjuncts, real morphology and real shared prefixes.
Random-article sampling from Wikipedia gives us that for free.
"""
import json
import os
import re
import sys
import time
import warnings

import requests

warnings.filterwarnings("ignore")

LANGS = {
    "en": 4_000_000,
    "hi": 4_000_000,
    "te": 3_500_000,
    "ta": 3_500_000,
    "bn": 3_000_000,
    "mr": 2_500_000,
    "kn": 2_500_000,
}

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
UA = "ERAv5-SpectralEmbedding-research/0.1 (buzzyperson@gmail.com)"

# Wikipedia extracts carry a lot of boilerplate section headers; strip the
# worst of it so the tokenizer is not trained on "== References ==".
BOILER = re.compile(r"^==+.*?==+$", re.M)
WS = re.compile(r"[ \t]+")


def clean(text: str) -> str:
    text = BOILER.sub("", text)
    text = WS.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) > 15]
    return "\n".join(lines)


def fetch_lang(lang: str, target_bytes: int) -> None:
    path = os.path.join(OUT, f"{lang}.txt")
    if os.path.exists(path) and os.path.getsize(path) >= target_bytes * 0.9:
        print(f"[{lang}] already have {os.path.getsize(path)} bytes", flush=True)
        return
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    url = f"https://{lang}.wikipedia.org/w/api.php"
    total = 0
    fails = 0
    mode = "a" if os.path.exists(path) else "w"
    total = os.path.getsize(path) if mode == "a" else 0
    with open(path, mode, encoding="utf-8") as fh:
        while total < target_bytes and fails < 40:
            try:
                r = sess.get(
                    url,
                    params={
                        "action": "query",
                        "generator": "random",
                        "grnnamespace": 0,
                        "grnlimit": 20,
                        "prop": "extracts",
                        "explaintext": 1,
                        "exlimit": 20,
                        "format": "json",
                    },
                    timeout=40,
                )
                pages = r.json().get("query", {}).get("pages", {})
            except Exception as exc:  # network flake, just retry
                fails += 1
                print(f"[{lang}] retry after {exc}", flush=True)
                time.sleep(3)
                continue
            got = 0
            for _pid, p in pages.items():
                txt = clean(p.get("extract", "") or "")
                if len(txt) < 60:
                    continue
                fh.write(txt + "\n")
                got += len(txt.encode("utf-8"))
            total += got
            fails = fails + 1 if got == 0 else 0  # only *consecutive* failures count
            fh.flush()
            print(f"[{lang}] {total:,} / {target_bytes:,}", flush=True)
            time.sleep(0.4)
    print(f"[{lang}] DONE {total:,} bytes -> {path}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(LANGS)
    for lang in only:
        fetch_lang(lang, LANGS[lang])
    sizes = {
        l: os.path.getsize(os.path.join(OUT, f"{l}.txt"))
        for l in LANGS
        if os.path.exists(os.path.join(OUT, f"{l}.txt"))
    }
    print(json.dumps(sizes, indent=2))
