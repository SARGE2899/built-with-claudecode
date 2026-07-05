"""
Top-p (Nucleus) Sampling.

Concept: top-k's weakness is that k is FIXED regardless of how the
probability mass is actually distributed -- k=2 might discard several
genuinely plausible tokens when the distribution is flat, or waste slots
on near-zero-probability tokens when the distribution is sharply peaked.
Top-p instead sorts tokens by probability descending and keeps the
SMALLEST prefix whose cumulative probability reaches at least p,
discarding the rest and renormalizing -- so the number of tokens kept
adapts automatically to how confident the distribution is at each step:
few tokens kept when one option dominates, more tokens kept when
probability is spread more evenly.

Uses the same hypothetical next-token logits as
`01_greedy_decoding.py` for the prompt "the cat sat on the ___".
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
NEXT_TOKEN_LOGITS = np.array([0.8, 0.0, -1.0, 0.5, 2.5])  # the, cat, sat, on, mat


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


full_probs = softmax(NEXT_TOKEN_LOGITS)
print("=== Top-p sampling internal state ===")
print("Prompt: 'the cat sat on the ___'  (same hypothetical logits as 01_greedy_decoding.py)")

order = np.argsort(-full_probs)
sorted_probs = full_probs[order]
cumulative = np.cumsum(sorted_probs)
print("\nSorted by probability, with running cumulative sum:")
for rank, idx in enumerate(order):
    print(f"  {ID_TO_WORD[idx]!r:>5s}: prob={full_probs[idx]:.4f}  "
          f"cumulative={cumulative[rank]:.4f}")


def top_p_filter(probs, p):
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    # keep every token up to and including the first one that reaches p
    cutoff = np.searchsorted(cumulative, p) + 1
    kept_idx = order[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[kept_idx] = probs[kept_idx]
    return filtered / filtered.sum(), kept_idx


rng = np.random.default_rng(0)
N_SAMPLES = 10

for p in [0.5, 0.9, 1.0]:
    filtered_probs, kept_idx = top_p_filter(full_probs, p)
    print(f"\n--- p = {p} ---")
    print(f"Kept tokens ({len(kept_idx)} of {len(VOCAB)}): {[ID_TO_WORD[i] for i in kept_idx]}")
    print("Renormalized probabilities (over kept tokens only):")
    for idx in kept_idx:
        print(f"  {ID_TO_WORD[idx]!r:>5s}: {filtered_probs[idx]:.4f}")
    samples = rng.choice(list(VOCAB.keys()), size=N_SAMPLES, p=filtered_probs)
    print(f"{N_SAMPLES} samples drawn: {[str(s) for s in samples]}")

print("\n=== Final output: top-k vs top-p, same distribution ===")
print("At p=0.5, only 'mat' is kept (its probability alone, 0.6992, "
      "already exceeds 0.5) -- top-p automatically kept just 1 token "
      "here, the same count `03_top_k_sampling.py` needed k=1 to force. "
      "At p=0.9, 'mat'+'the' (cumulative 0.8269) isn't quite enough, so "
      "'on' is included too, reaching 3 tokens -- top-p adapted its "
      "count to 3 without anyone choosing k=3 by hand. This is the "
      "core difference from top-k: the cutoff COUNT here is a "
      "consequence of the distribution's actual shape, not a fixed "
      "choice made in advance.")
