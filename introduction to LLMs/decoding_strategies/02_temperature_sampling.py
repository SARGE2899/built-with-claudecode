"""
Temperature Sampling.

Concept: instead of always taking the argmax (greedy), SAMPLE the next
token from the probability distribution -- giving lower-probability
tokens a genuine (if small) chance of being chosen, so the same prompt
can produce different continuations across runs. Temperature controls
how sharp or flat that distribution is before sampling, by dividing
every logit by T before the softmax:
  - T < 1 sharpens the distribution (pushes it toward greedy/argmax;
    T -> 0 recovers greedy exactly)
  - T = 1 leaves the original distribution unchanged
  - T > 1 flattens the distribution toward uniform (more randomness,
    more chance of a low-probability token getting picked)

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


print("=== Temperature sampling internal state ===")
print("Prompt: 'the cat sat on the ___'  (same hypothetical logits as 01_greedy_decoding.py)")

temperatures = [0.2, 1.0, 2.0]
rng = np.random.default_rng(0)
N_SAMPLES = 10

for T in temperatures:
    scaled_probs = softmax(NEXT_TOKEN_LOGITS / T)
    print(f"\n--- Temperature = {T} ---")
    print("Resulting probabilities:")
    for word, idx in VOCAB.items():
        print(f"  {word!r:>5s}: {scaled_probs[idx]:.4f}")
    samples = rng.choice(list(VOCAB.keys()), size=N_SAMPLES, p=scaled_probs)
    print(f"{N_SAMPLES} samples drawn: {[str(s) for s in samples]}")

print("\n=== Final output: comparing the 3 temperatures side by side ===")
print(f"{'word':>6s}  " + "  ".join(f"T={t}" for t in temperatures))
for word, idx in VOCAB.items():
    row = "  ".join(f"{softmax(NEXT_TOKEN_LOGITS / t)[idx]:.4f}" for t in temperatures)
    print(f"{word!r:>6s}  {row}")

print("\nNOTE: at T=0.2 (sharp), probability mass concentrates almost "
      "entirely on 'mat' -- sampling behaves nearly identically to "
      "greedy. At T=2.0 (flat), the gap between 'mat' and the other "
      "words shrinks substantially, and the sample list above should "
      "show noticeably more variety than at T=0.2. T=1.0 is the "
      "original, unmodified distribution -- exactly what "
      "`01_greedy_decoding.py` took the argmax of.")
