"""
Greedy Decoding.

Concept: the simplest possible way to turn a model's output distribution
into an actual next token -- just take the single highest-probability
(equivalently, highest-logit) token, every time, with no randomness at
all. Simple, fully deterministic (same input always gives the same
output), and cheap, but with two real weaknesses shown below: it can
never produce any of the OTHER plausible continuations a model
considered, and it's prone to getting stuck repeating itself if the
distribution ever comes to favor "repeat the last token" (a real,
well-documented degenerate failure mode of trained language models, not
just a toy artifact -- see `06_repetition_penalty.py` for the fix).

Every decoding script in this folder starts from the same HYPOTHETICAL
next-token logit vector for the prompt "the cat sat on the ___" (i.e.
predicting the word that completes the canonical sentence). These
logits are hand-specified to stand in for "whatever a trained model's
output layer produced at this point" -- they are NOT derived from an
actual trained model (this project never trains one), only chosen to
make each decoding strategy's mechanics clearly visible and comparable
against the others.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}

# Hypothetical logits for "the cat sat on the ___": 'mat' is the clear
# favorite (it's the true completion), 'the'/'on' are plausible-ish
# runners-up, 'cat'/'sat' are unlikely.
NEXT_TOKEN_LOGITS = np.array([0.8, 0.0, -1.0, 0.5, 2.5])  # the, cat, sat, on, mat


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


probs = softmax(NEXT_TOKEN_LOGITS)

print("=== Greedy decoding internal state ===")
print("Prompt: 'the cat sat on the ___'")
print("Hypothetical next-token logits and probabilities:")
for word, idx in VOCAB.items():
    print(f"  {word!r:>5s}: logit={NEXT_TOKEN_LOGITS[idx]:+.2f}  prob={probs[idx]:.4f}")

chosen_id = int(np.argmax(probs))
print(f"\n=== Final output: greedy choice ===")
print(f"argmax -> {ID_TO_WORD[chosen_id]!r} (prob={probs[chosen_id]:.4f})")
print("Deterministic: running this again with the exact same logits will "
      "always pick the exact same token, unlike every sampling-based "
      "strategy in this folder.")

print("\n=== Weakness demonstration: greedy can get stuck looping ===")
print("Suppose (hypothetically) that once the model has just generated "
      "'mat', its next-token logits come to strongly favor 'mat' again "
      "(a real degenerate pattern seen in under-trained/repetition-prone "
      "language models, not just a toy scenario):")
loop_logits = np.array([0.0, 0.0, -1.0, 0.0, 3.0])  # 'mat' dominates after 'mat'
loop_probs = softmax(loop_logits)
for word, idx in VOCAB.items():
    print(f"  {word!r:>5s}: logit={loop_logits[idx]:+.2f}  prob={loop_probs[idx]:.4f}")

generated = ["mat"]
for step in range(4):
    next_word = ID_TO_WORD[int(np.argmax(loop_probs))]
    generated.append(next_word)
print(f"Greedy-generated continuation: {generated}")
print("Every step picks the same argmax again -- greedy has no mechanism "
      "to ever break out of this once it's the highest-logit choice; "
      "`06_repetition_penalty.py` shows a direct fix for exactly this.")
