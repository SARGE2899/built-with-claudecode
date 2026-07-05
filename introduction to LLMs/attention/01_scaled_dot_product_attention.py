"""
Scaled Dot-Product (Self-)Attention.

Concept: for every position, ask "how relevant is every other position
to me right now?", turn those relevance scores into weights that sum to
1 (via softmax), then output a weighted average of every position's
"value" vector using those weights. Concretely: project the input into
three learned views -- Query (what am I looking for), Key (what do I
offer, to be matched against queries), and Value (what do I actually
contribute if attended to) -- score every query against every key via a
dot product, scale by 1/sqrt(d_k) (keeps scores from growing huge as
dimensionality increases, which would push softmax into saturated,
near-one-hot territory), softmax each row into weights, then combine
Values with those weights.

This is "self"-attention because Q, K, and V are all projections of the
SAME input sequence -- every position attends to every other position in
its own sequence (contrast with cross-attention, where queries come from
one sequence and keys/values from another).

Uses the combined token+positional embeddings from
`embeddings/03_combining_embeddings.py` as input, and the same d_model=8
every other script in this project shares.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)


def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None]
    dim = np.arange(d_model)[None, :]
    angle_rates = 1.0 / (10000 ** (2 * (dim // 2) / d_model))
    angles = position * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


rng = np.random.default_rng(0)
token_embedding_table = rng.normal(loc=0.0, scale=0.1, size=(len(VOCAB), D_MODEL))
X = token_embedding_table[token_ids] + sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)

# Learned projection matrices (randomly initialized -- untrained, as with
# every weight matrix elsewhere in this project).
W_q = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_k = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_v = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))

Q = X @ W_q
K = X @ W_k
V = X @ W_v

print("=== Scaled dot-product attention internal state ===")
print(f"Input X shape: {X.shape} (seq_len={SEQ_LEN}, d_model={D_MODEL})")
print(f"Q, K, V shapes: {Q.shape} each (one query/key/value vector per position)")

d_k = K.shape[-1]
raw_scores = Q @ K.T
scaled_scores = raw_scores / np.sqrt(d_k)

print(f"\nRaw attention scores (Q @ K^T), shape {raw_scores.shape}:")
print(np.round(raw_scores, 2))
print(f"\nScaled scores (divided by sqrt(d_k)={np.sqrt(d_k):.3f}):")
print(np.round(scaled_scores, 2))
print("NOTE: scaling matters more as d_k grows -- with d_k=8 the effect "
      "is modest here, but without it, larger d_k values would produce "
      "scores large enough to push softmax toward near-one-hot outputs "
      "almost everywhere, making gradients vanish during training.")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # numerical stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


attention_weights = softmax(scaled_scores, axis=-1)

print(f"\nAttention weights (softmax over each row -- each row sums to 1):")
header = "        " + "  ".join(f"{ID_TO_WORD[t]:>5s}" for t in token_ids)
print(header)
for pos, row in enumerate(attention_weights):
    print(f"  {ID_TO_WORD[token_ids[pos]]:>5s}: {np.round(row, 3)}  (sum={row.sum():.3f})")

output = attention_weights @ V

print(f"\n=== Final output: attention output per position (shape {output.shape}) ===")
for pos, vec in enumerate(output):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): {np.round(vec, 3)}")

print("\nNOTE: every output position is a weighted blend of ALL positions' "
      "Value vectors -- including positions that come AFTER it (e.g. "
      "position 0 attends to position 5). That's fine for encoder-style "
      "attention, but an autoregressive LLM generating text left-to-right "
      "cannot be allowed to look at future tokens it hasn't generated "
      "yet -- `02_causal_masking.py` shows the fix.")
