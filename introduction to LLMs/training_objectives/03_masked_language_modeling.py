"""
Masked Language Modeling (MLM) -- contrast with next-token prediction.

Concept: BERT-style models use a completely different training
objective than generative LLMs. Instead of predicting each token from
only what came BEFORE it (causal, left-to-right, as in
`01_next_token_prediction.py`), MLM randomly replaces some fraction of
input tokens with a special [MASK] placeholder and trains the model to
recover the ORIGINAL token at each masked position -- using context from
BOTH directions (words before AND after the mask), since there's no
causal mask restricting attention here at all.

This bidirectional context makes MLM excellent for building rich
representations of existing text (classification, similarity, etc.),
but it's fundamentally awkward for open-ended GENERATION: at generation
time you'd need to already know how many tokens come after the one
you're predicting, and what all-but-one of them are -- which is exactly
backwards from "generate text the model hasn't produced yet". This is
the core reason today's generative LLMs (GPT-family, Claude, etc.) are
trained with next-token prediction, not MLM, even though MLM produces
very strong bidirectional representations for other uses.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
MASK_ID = len(VOCAB)  # a new id, one past the real vocabulary
ID_TO_WORD = {**{i: w for w, i in VOCAB.items()}, MASK_ID: "[MASK]"}
VOCAB_SIZE = len(VOCAB)  # the model only ever needs to PREDICT real vocab tokens
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)

rng = np.random.default_rng(0)

# Randomly mask ~30% of positions (rounded up to at least 1), a smaller
# fraction than BERT's real 15% only because our sequence is so short
# that 15% of 6 tokens would round to 0.
n_to_mask = max(1, round(SEQ_LEN * 0.3))
masked_positions = sorted(rng.choice(SEQ_LEN, size=n_to_mask, replace=False).tolist())

print("=== Masking internal state ===")
print(f"Original sentence: {SENTENCE!r}")
print(f"Positions masked: {masked_positions} "
      f"(original tokens there: {[ID_TO_WORD[token_ids[p]] for p in masked_positions]})")

masked_input_ids = list(token_ids)
for p in masked_positions:
    masked_input_ids[p] = MASK_ID
print(f"Masked input:      {[ID_TO_WORD[i] for i in masked_input_ids]}")

# Embedding table needs one extra row for [MASK] itself, since it's a
# token the model must be able to embed and process like any other.
embedding_table = rng.normal(scale=0.1, size=(VOCAB_SIZE + 1, D_MODEL))
mlm_head = rng.normal(scale=0.1, size=(D_MODEL, VOCAB_SIZE))  # predicts only REAL vocab tokens

X = embedding_table[masked_input_ids]  # (seq_len, d_model) -- NO causal mask applied anywhere

print(f"\nEmbedded (masked) input shape: {X.shape}")
print("NOTE: unlike every attention script in `attention/`, there is NO "
      "causal mask here -- a position being predicted is free to attend "
      "to positions on BOTH sides of it, including ones after it in the "
      "sequence. (This script only demonstrates the masking + prediction-"
      "target setup; it doesn't re-run full self-attention, since the "
      "only difference from `attention/01_scaled_dot_product_attention.py` "
      "would be skipping the mask entirely.)")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


logits = X @ mlm_head
probs = softmax(logits, axis=-1)

print(f"\n=== Final output: predictions at MASKED positions only ===")
print("(MLM's loss is computed ONLY at masked positions -- unmasked "
      "positions are already visible as input, so there's nothing to "
      "predict there, unlike next-token prediction's loss at EVERY "
      "position)")
losses = []
for p in masked_positions:
    true_id = token_ids[p]
    loss = -np.log(probs[p, true_id])
    losses.append(loss)
    rounded = {w: round(float(probs[p, i]), 3) for w, i in VOCAB.items()}
    print(f"  position {p} (true token={ID_TO_WORD[true_id]!r}): "
          f"P(each word)={rounded}  loss={loss:.4f}")

print(f"\nAverage loss over the {len(masked_positions)} masked position(s): "
      f"{np.mean(losses):.4f}")

print("\n=== Side-by-side: what each objective's loss actually covers ===")
print(f"Next-token prediction ({SEQ_LEN - 1} positions, causal, "
      f"left-context only): loss computed at EVERY position")
print(f"Masked LM ({len(masked_positions)} of {SEQ_LEN} positions, "
      f"bidirectional context): loss computed ONLY at masked positions")
