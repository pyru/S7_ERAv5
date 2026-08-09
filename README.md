# Spectral Byte Embeddings

### Kronecker V2 — Problems 4 and 5, answered by a single substitution

> **Problem 4.** *"What is a REAL Fourier alternative of Kronecker? Why can't I represent
> each character like a Fourier wave, and just add them to make a word?"*
>
> **Problem 5.** *"Kronecker is forward deterministic. How do I make a reverse of this?
> If we can do this, then we can get rid of the final head as well. Then we can have a
> vocab of 1M as well without any issues."*

These are the same problem. A wave basis is what makes a code invertible, and
invertibility is what deletes the head. This is one construction that answers both.

---

## Submission

**Interactive write-up (live encoder, fold visualiser, every chart):**
https://claude.ai/code/artifact/8bcac6be-3c62-496d-8034-28ac53ffdb06

**Problems solved:** 4 (a real Fourier alternative) and 5 (running the codec backwards
to delete the output head). Problem 3 — the fixed 32-byte window — falls out as a
corollary and is measured alongside them.

**How it is proved.** Five experiments on a *real* 16 MB multilingual Wikipedia corpus
with *real* byte-level BPE tokenizers trained on it, plus a small transformer trained
five ways with a matched trunk, matched data and matched seed:

| | question | file | headline |
|---|---|---|---|
| E1 | does the window destroy real vocabulary? | `e1_collisions.py` | Kronecker collides on 16 673 words (6.99 %); spectral on **0** at the same budget |
| E2 | does the code invert? | `e2_invertibility.py` | exact round-trip in-window for every script |
| E3a | does the inverse survive noise? | `e2_invertibility.py` | null result — both codecs immune to σ = 1.0 |
| E3b | how wide must `d_model` be? | `e2_invertibility.py` | the 2 048-dim code is lossless at `d_model` = 512 |
| E4 | does a head-free model learn language? | `e4_lm.py` | works, head = **1 parameter**, and costs 0.6 bits/byte at `d_model` 256 |
| E5 | is a 1M vocabulary really free? | `e5_vocab_scaling.py` | 524 289 params at V = 32 k, 131 k **and 1 M**, identical step time |

**Start here:**

```bash
python spectral/verify.py      # re-checks every claim in this README, in seconds
```

Then read §3 below, or open the link above and type a Tamil word into the dial.

**Three results that went against me**, kept in rather than tidied away: §3.2 (a
negative result on BPE vocabularies), §3.6 (spectral is *not* a better language model
than Kronecker — the two are tied), and §3.6 again (a prediction I made from E3b that
E4 falsified).

---

## 0. The idea in three lines

The shipped Kronecker codec builds a token's code as a sum of outer products of two
**one-hot** vectors:

```
kappa(w) = (1/sqrt L) * SUM_p  onehot256(byte_p)  (x)  onehot32(p)
```

Spectral Byte Embedding keeps the skeleton and replaces the position basis with a
**bank of waves**:

```
S(w)     = (1/sqrt L) * SUM_p  phi(byte_p)  (x)  psi(p)

psi(p) = [ .. cos(w_k p), sin(w_k p) .. ]     k = 0 .. K-1
```

Every character contributes a wave packet; the word is their superposition. That is
Problem 4, literally. And because `psi` is a known linear map, the code runs backwards
through a fixed matched filter with **zero trainable parameters** — that is Problem 5.

---

## 1. Why swapping the basis changes anything

For any token shorter than the window the two codes carry *identical* information —
`psi` is an orthogonal change of basis on the very same grid. The difference appears
only past the window, and there it is total.

| | one-hot position (Kronecker) | wave position (spectral) |
|---|---|---|
| p ≥ 32 | there is no column — the byte is **multiplied by zero** | `psi(p)` is periodic — the byte **folds onto p mod 32 and adds** |
| failure | information **destroyed** | information **superimposed** |
| when two tokens collide | whenever they share a 32-byte prefix | only when their folded sums coincide |

That distinction is the whole paper. Truncation is the special case of a fold whose
out-of-window term was replaced by zero, so **folding strictly dominates truncation**:
it never discards anything truncation keeps, and it keeps things truncation discards.

It matters because the two failure modes have completely different relationships to
language. A shared prefix is not a rare accident — shared prefixes are what
morphology *is*. Truncation's failure mode is aligned with the data. Folding's is not:
a collision needs a numerical coincidence, not a spelling pattern.

---

## 2. Running it backwards, and deleting the head

Because the position basis is an orthonormal real-DFT frame, the inverse is a matched
filter — a single fixed tensor contraction:

```
charlogits[p, b]  =  < code , phi(b) (x) psi(p) >
```

This is **exact** for any token that fits the grid, and it has no parameters. So the
output path of the model becomes:

```
pred_code = h @ W          # W is the INPUT projection, reused transposed
charlogits = fixed_inverse(pred_code)
loss = cross_entropy over the target token's BYTES
```

There is no `d_model x |V|` matrix anywhere in the graph. The only trainable tensor
facing the vocabulary is `W`, of shape `code_dim x d_model` — and **`|V|` does not
appear in that shape**. The measured parameter count for the head in the language
model below is literally **1** (a scalar temperature).

Three consequences, all measured in §3:

- Vocabulary size stops costing parameters *on both sides* of the model.
- The model emits a **string**, not a vocabulary index, so it can produce tokens that
  were never in the vocabulary.
- The output-path FLOPs become `O(code_dim)` instead of `O(|V|)`.

---

## 3. What was measured

Everything below is measured on a **real** 16 MB multilingual Wikipedia corpus
(hi 4.0 MB, ta 3.5 MB, bn 3.0 MB, te 2.1 MB, kn 1.9 MB, en 1.2 MB, mr 0.2 MB) with
**real** byte-level BPE tokenizers trained on it at 8 192 / 32 000 / 131 072.
Nothing here is synthetic.

### 3.1 The 32-byte window is a sovereignty bug — at word level

The corpus's own distinct words, by share that overflows a 32-byte window:

| en | hi | bn | te | kn | ta |
|---|---|---|---|---|---|
| **0.05 %** | 6.45 % | 14.31 % | 19.21 % | 31.41 % | **35.06 %** |

Encoding all 238 584 distinct words and counting tokens that become **bit-identical**:

| codec | code dim | tokens collided | groups | decode exact |
|---|---:|---:|---:|---:|
| Kronecker `pos_dim=32` (shipped) | 8 192 | **16 673 (6.99 %)** | 5 445 | 0.811 |
| Kronecker `pos_dim=48` | 12 288 | 1 218 (0.51 %) | 478 | 0.974 |
| Kronecker `pos_dim=64` | 16 384 | 48 (0.02 %) | 19 | 0.997 |
| **Spectral DFT, 32 channels** | **8 192** | **0** | **0** | 0.810 |
| Spectral vernier3, 32 ch | 8 192 | **0** | 0 | 0.709 |
| Spectral vernier3, 64 ch | 16 384 | **0** | 0 | 0.996 |

**Zero collisions at half the budget Kronecker needs to get to 48.** And the damage
Kronecker takes is not distributed evenly — it is distributed by script:

| script | words collided under Kronecker-32 |
|---|---|
| Tamil | 8 475 (**18.5 %**) |
| Kannada | 3 824 (11.0 %) |
| Telugu | 1 965 (7.2 %) |
| Bengali | 1 845 (5.2 %) |
| Devanagari | 549 (1.4 %) |
| **Latin** | 15 (**0.03 %**) |

A 600× disparity between Tamil and Latin, from a constant in a config file. The
cleanest single example the census turned up:

```
இருப்பினும்      (33 bytes)
இருப்பினும்,     (34 bytes)      <- the same word, plus a comma

Kronecker-32 code difference : 0.00e+00      (bit-identical, forever)
Spectral  code difference    : 0.0811
```

Under the shipped codec a model cannot learn that one of those ends a clause.

### 3.2 An honest negative result: BPE tokens do not hit the wall

The same census over the actual BPE vocabularies found **zero collisions for every
codec at every vocabulary size**, because BPE tokens are short — BPE stops merging
when frequency runs out:

| vocab | max token bytes | share over 32 B |
|---|---:|---:|
| 8 192 | 16 | 0.00 % |
| 32 000 | 25 | 0.00 % |
| 131 072 | 40 | 0.05 % (Bengali only) |

So the strong form of the Session 7 §8 claim — that the shipped `pos_dim=32` is
silently corrupting the V5 BPE vocabulary — **is not supported on this corpus**, and
I am reporting that rather than burying it. Two qualifications keep it from settling
the question:

1. The trend is monotone and points the right way: max token length runs
   16 → 25 → **40** bytes as the vocabulary goes 8 k → 32 k → 131 k. A 16 MB corpus
   only produced ~85 k real merges at the 131 072 setting, so this proxy *understates*
   what a production corpus does at that vocabulary.
2. Word level is where a byte codec earns its keep anyway. The advertised benefit of
   a byte codec is that unseen strings still get a sensible vector — and unseen
   strings are words, not BPE tokens. That regime is §3.1, and there the wall is
   brutal.

### 3.3 Round-trip fidelity against token length

Exact byte-for-byte recovery on real words, by length bucket:

| codec | 1–8 | 9–16 | 17–24 | 25–32 | 33–40 | 41–56 | 57–80 |
|---|---|---|---|---|---|---|---|
| Kronecker-32 (8 192) | 1.00 | 1.00 | 1.00 | 1.00 | **0** | **0** | **0** |
| Kronecker-64 (16 384) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.795 |
| Spectral DFT-32 (8 192) | 1.00 | 1.00 | 1.00 | 0.994 | 0 | 0 | 0 |
| Spectral vernier3-32 (8 192) | 1.00 | 1.00 | 0.950 | 0.429 | 0.001 | 0.001 | 0 |
| Spectral vernier3-64 (16 384) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.982 | **0.872** |

Two things worth separating, because conflating them is easy:

- **Collisions and decodability are different questions.** Spectral DFT-32 has *zero*
  collisions yet only decodes 81 % of words. Past 32 bytes its codes stay distinct
  (folded, not destroyed) but are no longer *resolvable position by position*. For an
  embedding, distinctness is what matters — that is the §3.1 result. For head-free
  generation, decodability is what matters, and there the channel count binds both
  designs.
- **A multi-modulus ("vernier") ladder buys range and costs exactness.** On *random*
  bytes it decodes 80 positions from 32 channels — genuine super-resolution via a
  Chinese-remainder argument. On *real text* it degrades, because real UTF-8 repeats
  continuation bytes heavily and breaks the sparsity its recovery assumes. At 32
  channels it is worse than plain DFT; at 64 channels it beats Kronecker-64 on long
  words (0.872 vs 0.795 at 57–80 bytes) with zero collisions.

**Design conclusion: use the plain DFT ladder by default** — it is exactly orthonormal
(condition number 1.000, verified), exactly invertible in-window, and collision-free.
Reach for vernier only when long-token *decoding* matters more than in-window
exactness.

This was not the ladder I started with. A geometric/RoPE-style ladder is the obvious
first guess and it is **broken**: `w = pi` makes `sin(pi p) = 0` for every integer `p`,
a dead channel, and the slow channels cannot separate adjacent positions. Measured
condition number `3.3e17`. The ladder is the design, not a detail.

### 3.4 Noise: a null result

Additive Gaussian error scaled to each code's own RMS, decoded with the fixed inverse.
Both designs hold at **1.000 exact recovery all the way to σ = 1.0**; only the 2048-dim
spectral code shows any decay at all (0.9895 at σ = 1.0). There is no difference to
report here, and reporting it as a difference would be wrong.

Both survive for different reasons, which is worth a sentence: a one-hot peak has an
enormous margin over noise spread across 8 192 cells, while the spectral matched filter
integrates all 8 192 channels coherently, buying back a factor of √8192 in SNR.

An earlier version of this experiment showed Kronecker collapsing at σ = 0.05. That was
a bug in **my** baseline decoder — it inferred token length from an absolute energy
threshold, so under noise every column read as "live" and the length was always wrong.
Comparing against a threshold relative to the token's own peak fixes it. The corrected
baseline is the one reported.

### 3.5 The rank bottleneck — the measurement that decides the design

The model emits `d_model` numbers, so reachable codes lie in a `d_model`-dimensional
subspace. Compressing the whole 32 k vocabulary's codes to `d` dimensions by PCA — the
best any linear projection can do — then reconstructing and decoding:

| dims kept | 32 | 128 | **256** | 384 | **512** | 768 | 1024 | 1536 |
|---|---|---|---|---|---|---|---|---|
| Kronecker 8 192 | .007 | .191 | **.485** | .705 | .825 | .948 | .980 | 1.000 |
| Spectral 8 192 | .016 | .305 | **.621** | .798 | .891 | .977 | .993 | 1.000 |
| **Spectral 2 048** | .049 | .593 | **.944** | .993 | **1.000** | 1.000 | 1.000 | — |

Two results:

- **The wave basis is uniformly more compressible than the one-hot basis** at identical
  code size — 62.1 % vs 48.5 % at `d` = 256 — because it concentrates the vocabulary's
  variance into fewer principal directions.
- **The compact spectral code is lossless at `d_model` = 512.** So the design is not
  merely "use waves", it is "use waves *and* a compact character basis". At V5's
  `d_model` = 8 096 the bottleneck does not exist; the toy at 256 sits just below the
  knee, which is a deliberately pessimistic proxy.

### 3.6 Does a head-free model actually learn language?

Five arms, identical trunk, identical data, identical seed, so every adjacent pair
isolates one variable. 1 000 steps, `d_model` 256, 4 layers, V = 32 000.

| arm | input | head | input params | head params | total | **val bits/byte** | exact next-token |
|---|---|---|---:|---:|---:|---:|---:|
| A | dense table | dense untied | 8 192 000 | 8 192 000 | 19 534 336 | 1.4748 | 24.9 % |
| B | Kronecker 8 192 | dense | 2 097 152 | 8 192 000 | 13 439 488 | **1.4323** | 25.8 % |
| C | spectral 8 192 | dense | 2 097 152 | 8 192 000 | 13 439 488 | **1.4334** | 25.5 % |
| D | spectral 8 192 | **none** | 2 097 152 | **1** | 5 247 489 | 2.0919 | 17.5 % |
| E | spectral 2 048 | **none** | 524 288 | **1** | 3 674 625 | 2.5735 | 9.4 % |

The metric is **bits per byte**, not perplexity, because the arms do not agree on what
a token costs to emit. It is *biased against* the head-free arms: they define a
distribution over all byte strings, including strings that are not tokens, so their
BPB is an upper bound on a vocabulary-renormalised version.

Three readings, including one that went against me:

- **The byte codecs beat the dense table.** B at 1.4323 and C at 1.4334 against the
  control's 1.4748, with 6.1 M fewer parameters each. B and C are tied to within noise
  — so at equal budget **a spectral code is not a better language model than a
  Kronecker one**. Its advantages are the ones in §3.1 and §3.5: no collisions, and an
  analytic inverse. Not perplexity.
- **Deleting the head costs real quality at this scale.** D reaches 2.0919 against
  1.4748, with a head of *one* parameter and 3.7× fewer parameters overall. It still
  reproduces the next token's complete byte string exactly **17.5 %** of the time with
  no vocabulary consulted anywhere — a working mechanism, not a free lunch. The gap is
  what §3.5 predicts: at `d_model` = 256 only 62 % of the 8 192-dim codes survive a
  linear squeeze to 256 dimensions. V5's `d_model` = 8 096 sits far above that knee,
  but **that is an extrapolation from the rank curve, not something this run tested.**
- **A prediction of mine that failed.** §3.5 says the 2 048-dim code survives a 256-dim
  bottleneck far better (94 % vs 62 %), so I expected E to beat D. It did not — E is
  clearly worse (2.5735 vs 2.0919). The likely reason: the rank curve measures how well
  a code survives *compression*, whereas the trained decoder is limited by the capacity
  of the projection itself, and E's is a quarter of D's; E's 64-dim character basis is
  also lossy over 256 byte values where D's is exact. The rank curve bounds what is
  *possible*; it does not predict what is *learned*.

### 3.7 A vocabulary of one million

At V5's reference shape (`|V|` = 131 072, `d_model` = 8 096), the two token-facing
matrices cost:

| scheme | parameters | training memory |
|---|---:|---:|
| dense table, untied head | 2 122 317 824 | 33.96 GB |
| dense table, tied head | 1 061 158 912 | 16.98 GB |
| factorized r=512 + dense head | 1 132 412 928 | 18.12 GB |
| Kronecker 8 192 + dense head | 1 127 481 344 | 18.04 GB |
| **spectral 8 192, no head** | **66 322 432** | **1.06 GB** |

Note where the other compressions leak: factorization and Kronecker both shrink the
*input* table and then hand almost all of it back through a dense `d_model x |V|` head.
Only deleting the head removes both.

The code is a pure function of a token's bytes, so the `|V| x code_dim` table is a
convenience, never a requirement. Above a threshold the implementation computes codes
on demand for the distinct tokens in the batch and never materialises the table at all.
Building a real head-free model at three vocabulary sizes and running genuine
forward/backward passes on this CPU:

| vocabulary | token-interface params | head params | sec/step | dense untied would cost | reduction |
|---:|---:|---:|---:|---:|---:|
| 32 000 | 524 289 | 1 | 1.187 | 16 384 000 | 31× |
| 131 072 | 524 289 | 1 | 1.172 | 67 108 864 | 128× |
| **1 000 000** | **524 289** | **1** | **1.202** | 512 000 000 | **977×** |

The parameter count is *identical* across all three rows and so is the step time, to
within noise. `|V|` does not appear in the cost of the token interface, on either side
of the model. That is the whole of the "1M vocab without any issues" claim, and it is
measured rather than projected.

---

## 4. Honest limitations

- **The likelihood is byte-factorised.** `p(token) = prod_p p(byte_p | h)` ignores
  correlations between bytes inside a token, and puts mass on strings that are not
  tokens. Reported BPB is therefore an upper bound. A trie-constrained
  renormalisation over the vocabulary would make it exact and is not implemented here.
- **Training is a proxy, not a converged result.** CPU-only, ~3.7 M–19.5 M parameters,
  ~1 M tokens. The arms are matched in data, seed and trunk, so the *ordering* is
  meaningful; the absolute numbers are not.
- **The corpus is 16 MB.** Large enough for a real multilingual BPE and a real word
  census, too small to reproduce what a production corpus does to token lengths at
  V=131 072 (see §3.2).
- **Past the grid, decodability is gone, not just degraded.** Folding preserves
  *distinctness* past 32 bytes, which is what the embedding needs; it does not
  preserve *resolvability*, which is what generation needs. For generation the channel
  count has to cover the longest token you intend to emit.
- **The rank bottleneck is real** and is why §3.4 exists. At V5's reference shape
  (`code_dim` 8 192, `d_model` 8 096) the ratio is ≈1.01 and the bottleneck
  essentially vanishes; in the toy it is 8–32×, so the toy is a *pessimistic* proxy.

---

## 5. What this would change in the V5 embedding decision

| Session 7 decision | what this work supports |
|---|---|
| `pos_dim` chosen by collision count, starting from 32 | the count is **0** on BPE tokens and **6.99 %** on words. If the input path only ever sees BPE tokens, 32 is fine and this is a non-issue. If it must handle unseen strings — which is the reason to have a byte codec at all — 32 is not defensible |
| Kronecker codec + trainable projection | swap the position basis for a DFT ladder. Same cost, same LM quality (1.4334 vs 1.4323 bpb), zero collisions instead of 16 673, and it inverts |
| output head **untied** | unchanged for now. Tying via the codec is *exact* rather than heuristic, but the head-free arm cost 0.6 bits/byte at `d_model` 256. Worth a proxy run at `d_model` ≥ 2048 before adopting; not worth adopting on this evidence |
| `embedding_policy_id` in the ledger | add `ladder` and `pos_ch`. A geometric ladder is **silently singular** (condition number 3.3e17) and a checkpoint should be able to say which ladder produced it |

The claim I would actually defend at this point: **the spectral codec is a free swap for
Kronecker** — identical budget, identical LM quality, strictly fewer collisions, plus an
analytic inverse you can choose to use later. The head-free decoder is a working
mechanism with an enormous parameter win that has not yet been shown to be free at
realistic width.

---

## 6. Reproducing

```bash
python spectral/fetch_corpus.py                      # ~16 MB of Wikipedia, 7 languages
python spectral/vocab.py 32000                       # train + inspect a BPE vocabulary

python spectral/experiments/e1_collisions.py 8192 32000 131072   # -> results/e1_collisions.json
python spectral/experiments/e2_invertibility.py                  # -> results/e2_invertibility.json
STEPS=1000 python spectral/experiments/e4_lm.py                  # -> results/e4_lm.json
python spectral/experiments/e5_vocab_scaling.py                  # -> results/e5_vocab_scaling.json
```

Requires `torch`, `numpy`, `tokenizers`, `requests`. Runs on CPU; the full suite is a
few hours on 8 cores.

| file | what it holds |
|---|---|
| `spectral/codecs.py` | both codecs, the frequency ladders, the fixed inverse |
| `spectral/model.py` | small transformer with pluggable input path and head |
| `spectral/vocab.py` | BPE training, byte recovery, per-script classification |
| `spectral/data.py` | tokenized stream and the matched-batch sampler |
| `spectral/verify.py` | self-check of every claim in this README (seconds, no corpus) |
| `spectral/fetch_corpus.py` | the Wikipedia pull that produced `data/` |
| `spectral/experiments/` | the four experiments, each writing one JSON |
| `results/*.json` | every number quoted here, as produced |
| `webapp/build.py` | regenerates the page **from** `results/*.json` |
| `webapp/index.html` | the built page — no hand-copied numbers anywhere in it |

Regenerate the page after any experiment: `python webapp/build.py`.

---

## 7. If this became a paper

The result that would carry it is §3.1 together with §3.7: a byte codec whose failure
mode is *aligned with the scripts a sovereign model exists to serve*, replaced by one
whose cost is identical and whose failure mode is not, plus an output interface whose
size is independent of the vocabulary. The three honest negatives in §3.2 and §3.6 are
what the paper would have to be built around rather than against — the spectral codec
is a free swap, not a better language model, and the head-free decoder is not yet shown
free at realistic width.

The two experiments I would run next, in order:

1. **The head-free arm at `d_model` ≥ 2 048.** §3.5 says the bottleneck disappears there
   and §3.6 says that is exactly where the cost came from. This is the single
   measurement that decides whether the head can actually go.
2. **Trie-constrained renormalisation** over the vocabulary, to turn the byte-factorised
   likelihood into an exact one and remove the upper-bound caveat from every
   bits-per-byte number reported here.
