"""
Next-Token Prediction (Autoregressive Language Modeling).

Concept: the training objective behind every modern generative LLM --
at every position i, predict the token at position i+1, given only
tokens 0..i (matching the causal masking from
`attention/02_causal_masking.py`). A linear "language modeling head"
projects each position's d_model-dimensional vector to vocab_size
logits, softmax turns those into a probability distribution over the
whole vocabulary, and cross-entropy loss measures how much probability
mass landed on the actual next token (loss = -log(P(correct token))
-- 0 for a perfect confident correct prediction, growing arbitrarily
large as confidence in the WRONG answer increases).

Critically, this happens IN PARALLEL at every position in one forward
pass during training (position 0 predicts position 1, position 1
predicts position 2, ... all at once) -- it's only at GENERATION time
that prediction becomes genuinely sequential, one token at a time (see
`decoding_strategies/`).

Uses a random (untrained) embedding table + LM head at the same
d_model=8 as every other script -- this demonstrates the LOSS
COMPUTATION mechanics, not a trained model's actual predictions.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]

# Inputs are all tokens except the last; targets are all tokens shifted
# one to the right -- input[i] predicts target[i] = the ACTUAL next token.
inputs = token_ids[:-1]   # the, cat, sat, on, the
targets = token_ids[1:]   # cat, sat, on, the, mat
SEQ_LEN = len(inputs)

print("=== Next-token prediction internal state ===")
print(f"Full sentence tokens: {token_ids}")
print(f"Inputs  (positions 0..4): {inputs} -> {[ID_TO_WORD[i] for i in inputs]}")
print(f"Targets (positions 1..5): {targets} -> {[ID_TO_WORD[i] for i in targets]}")
print("(input[i] is used to predict target[i], the token that actually "
      "came right after it in the real sentence)")

rng = np.random.default_rng(0)
embedding_table = rng.normal(scale=0.1, size=(VOCAB_SIZE, D_MODEL))
lm_head = rng.normal(scale=0.1, size=(D_MODEL, VOCAB_SIZE))  # projects back to vocab_size logits

X = embedding_table[inputs]                 # (seq_len, d_model)
logits = X @ lm_head                        # (seq_len, vocab_size)

print(f"\nEmbedded inputs shape: {X.shape}")
print(f"LM head shape: {lm_head.shape} (d_model -> vocab_size, mapping back to token space)")
print(f"Logits shape: {logits.shape}")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


probs = softmax(logits, axis=-1)

print("\n=== Predicted next-token distribution at each position ===")
for pos in range(SEQ_LEN):
    rounded = {w: round(float(p), 3) for w, p in zip(VOCAB, probs[pos])}
    print(f"  input={ID_TO_WORD[inputs[pos]]!r:>6s} -> true next={ID_TO_WORD[targets[pos]]!r:>6s}  "
          f"P(each word)={rounded}")

print("\n=== Final output: cross-entropy loss per position ===")
losses = -np.log(probs[np.arange(SEQ_LEN), targets])
for pos in range(SEQ_LEN):
    print(f"  input={ID_TO_WORD[inputs[pos]]!r:>6s} -> true next={ID_TO_WORD[targets[pos]]!r:>6s}  "
          f"P(true next)={probs[pos, targets[pos]]:.4f}  loss={losses[pos]:.4f}")

print(f"\nAverage loss across all 5 positions: {losses.mean():.4f}")
print(f"(For reference, a UNIFORM random guess over {VOCAB_SIZE} tokens gives "
      f"loss = -log(1/{VOCAB_SIZE}) = {-np.log(1 / VOCAB_SIZE):.4f} -- this "
      f"untrained model's loss is in the same ballpark, as expected before "
      f"any training has adjusted the embedding table or LM head away from "
      f"their random initialization.)")
