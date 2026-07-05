"""
Top-k Sampling.

Concept: temperature sampling still lets even the LOWEST-probability
token be chosen occasionally, however rare -- which can occasionally
produce a genuinely bad, incoherent continuation. Top-k restricts
sampling to only the k highest-probability tokens, discards everything
else entirely (zero chance, not just low chance), and renormalizes
the remaining k probabilities to sum to 1 before sampling. k=1 is
identical to greedy decoding; k=vocab_size is identical to plain
(unrestricted) sampling.

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
print("=== Top-k sampling internal state ===")
print("Prompt: 'the cat sat on the ___'  (same hypothetical logits as 01_greedy_decoding.py)")
print("\nFull distribution, sorted by probability:")
order = np.argsort(-full_probs)
for idx in order:
    print(f"  {ID_TO_WORD[idx]!r:>5s}: {full_probs[idx]:.4f}")


def top_k_filter(probs, k):
    order = np.argsort(-probs)
    top_k_idx = order[:k]
    filtered = np.zeros_like(probs)
    filtered[top_k_idx] = probs[top_k_idx]
    return filtered / filtered.sum(), top_k_idx


rng = np.random.default_rng(0)
N_SAMPLES = 10

for k in [1, 2, 5]:
    filtered_probs, kept_idx = top_k_filter(full_probs, k)
    print(f"\n--- k = {k} ---")
    print(f"Kept tokens: {[ID_TO_WORD[i] for i in kept_idx]}")
    print("Renormalized probabilities (over kept tokens only):")
    for idx in kept_idx:
        print(f"  {ID_TO_WORD[idx]!r:>5s}: {filtered_probs[idx]:.4f}")
    samples = rng.choice(list(VOCAB.keys()), size=N_SAMPLES, p=filtered_probs)
    print(f"{N_SAMPLES} samples drawn: {[str(s) for s in samples]}")

print("\n=== Final output: what k controls ===")
print("k=1  -> identical to greedy decoding (only 'mat' is ever possible)")
print("k=2  -> 'mat' and 'the' only; 'cat'/'sat'/'on' can now NEVER be "
      "chosen, no matter how many times you sample")
print("k=5 (=vocab_size) -> nothing is filtered out; identical to plain "
      "T=1.0 sampling in `02_temperature_sampling.py`")
print("\nNOTE: unlike temperature, top-k draws a hard line -- a token "
      "ranked (k+1)-th has EXACTLY zero probability, however close its "
      "score was to the k-th token's. This is precisely what prevents "
      "the rare 'sample the worst possible token' outcome temperature "
      "sampling can never fully rule out.")
