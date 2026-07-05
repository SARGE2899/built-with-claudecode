"""
Multi-Head Attention.

Concept: rather than computing one attention pattern using the full
d_model-dimensional Q/K/V, split each into `n_heads` smaller subspaces
(head_dim = d_model / n_heads) and run scaled dot-product attention
independently, IN PARALLEL, within each subspace. Each head can then
specialize in attending to different kinds of relationships (e.g. one
head might learn to track adjacent-word patterns, another longer-range
dependencies) using its own slice of the representation. The heads'
outputs are concatenated back to d_model and passed through one final
output projection (W_o) that lets the model recombine/mix information
across heads.

Uses n_heads=2 (so head_dim=8/2=4), the same causal masking from
`02_causal_masking.py`, and the same X input as the other attention
scripts.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
N_HEADS = 2
HEAD_DIM = D_MODEL // N_HEADS
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


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


rng = np.random.default_rng(0)
token_embedding_table = rng.normal(loc=0.0, scale=0.1, size=(len(VOCAB), D_MODEL))
X = token_embedding_table[token_ids] + sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)

W_q = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_k = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_v = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_o = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))

Q, K, V = X @ W_q, X @ W_k, X @ W_v

print("=== Multi-head attention internal state ===")
print(f"d_model={D_MODEL}, n_heads={N_HEADS}, head_dim={HEAD_DIM}")

# Split the last dimension into (n_heads, head_dim): each head gets its
# own contiguous slice of Q/K/V, not a learned split -- the split itself
# is just a reshape, all the "learning where to specialize" happens via
# W_q/W_k/W_v producing useful values in each slice during training.
def split_heads(x):
    seq_len = x.shape[0]
    return x.reshape(seq_len, N_HEADS, HEAD_DIM).transpose(1, 0, 2)  # (n_heads, seq_len, head_dim)


Q_heads, K_heads, V_heads = split_heads(Q), split_heads(K), split_heads(V)
print(f"Q split into per-head shape: {Q_heads.shape} (n_heads, seq_len, head_dim)")

causal_mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)

head_outputs = []
for h in range(N_HEADS):
    scores = (Q_heads[h] @ K_heads[h].T) / np.sqrt(HEAD_DIM)
    scores = np.where(causal_mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    out = weights @ V_heads[h]
    head_outputs.append(out)
    print(f"\n--- Head {h} ---")
    print(f"Attention weights (causally masked):")
    for pos, row in enumerate(weights):
        print(f"  {ID_TO_WORD[token_ids[pos]]:>5s}: {np.round(row, 3)}")

concatenated = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads * head_dim) = (seq_len, d_model)
print(f"\nConcatenated head outputs shape: {concatenated.shape} "
      f"(back to d_model={D_MODEL} by stacking all {N_HEADS} heads' outputs side by side)")

output = concatenated @ W_o

print(f"\n=== Final output: multi-head attention output (shape {output.shape}) ===")
for pos, vec in enumerate(output):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): {np.round(vec, 3)}")

print("\nNOTE: compare the two heads' attention weight tables above -- "
      "they operate on different 4-dimensional slices of the same Q/K/V "
      "projections, so even with the identical causal mask applied to "
      "both, each head distributes its attention differently across "
      "positions. W_o then mixes the two heads' 4-dim outputs back "
      "together into one 8-dim vector per position, giving later layers "
      "access to whatever each head separately noticed.")
