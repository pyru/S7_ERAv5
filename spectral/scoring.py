"""Exact vocabulary-renormalised likelihood for a head-free model.

WHY THIS EXISTS
---------------
A head-free model defines a distribution over *byte strings*:

    p_bytes(w) = prod_p  p(byte_p | h)

Most byte strings are not tokens, so p_bytes puts mass outside the vocabulary
and sum_{v in V} p_bytes(v) < 1.  Scoring a token with p_bytes therefore
charges the model for probability it spent on strings the dense-head arms never
had to consider, and every bits-per-byte number computed that way is an UPPER
BOUND on the model's real cost.  This module removes that handicap:

    p_V(v) = p_bytes(v) / sum_{u in V} p_bytes(u)

which is the distribution the dense head is scored under, so the two become
directly comparable.

THE COST, STATED HONESTLY
-------------------------
The normaliser sums over the whole vocabulary, so computing it is O(|V|) --
exactly the cost the head-free design was meant to avoid.  That is not a
contradiction, because the O(|V|) term appears in only one of the three things
a model does:

  training    factorised cross-entropy over bytes      O(code_dim)   no |V|
  generation  sample bytes, or walk a vocabulary trie  O(branching)  no |V|
  scoring     exact perplexity over a fixed vocabulary O(|V|)        <-- here

Only exact likelihood *evaluation* needs it, and evaluation is not a deployment
cost. It is used here to make an honest comparison, not because a deployed
model would run it.
"""
from __future__ import annotations

import torch


class VocabScorer:
    """Exact log p(token) under a byte-factorised model, renormalised over V."""

    def __init__(self, byte_targets: torch.Tensor):
        """byte_targets: [V, G] from model.byte_targets, -100 past token end."""
        self.mask = (byte_targets != -100).float()          # [V, G]
        self.index = byte_targets.clamp_min(0)              # [V, G], safe to gather
        self.V, self.G = byte_targets.shape

    @torch.no_grad()
    def log_probs(self, char_logits: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """[N, G, 256] char logits -> [N, V] exact renormalised log p(token).

        Chunked over N because the intermediate is [chunk, G, V]; at |V| = 32k
        and G = 32 that is 65 M floats per 64 positions, which is the largest
        tensor in the evaluation.
        """
        N = char_logits.shape[0]
        out = torch.empty(N, self.V)
        idx = self.index.T.contiguous()                     # [G, V]
        for s in range(0, N, chunk):
            ls = torch.log_softmax(char_logits[s : s + chunk], dim=-1)   # [n,G,256]
            n = ls.shape[0]
            # gather[n, g, v] = log p(byte = index[v, g] at position g)
            g = torch.gather(ls, 2, idx.unsqueeze(0).expand(n, self.G, self.V))
            lp = (g * self.mask.T.unsqueeze(0)).sum(dim=1)   # [n, V]
            out[s : s + chunk] = lp - lp.logsumexp(dim=-1, keepdim=True)
        return out

    @torch.no_grad()
    def nll(self, char_logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Total exact negative log-likelihood (nats) of the target tokens."""
        lp = self.log_probs(char_logits)
        return float(-lp.gather(1, targets.view(-1, 1)).sum())
