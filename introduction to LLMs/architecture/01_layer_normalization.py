"""
Layer Normalization.

Concept: deep networks are prone to activations drifting to very large
or very small scales layer after layer, which destabilizes training.
Layer Normalization fixes the SCALE at every position independently: for
each position's feature vector, subtract its own mean and divide by its
own standard deviation (across the d_model features, NOT across the
batch or sequence -- this is what distinguishes it from BatchNorm, and
is why it works even with a sequence length of 1 or a batch size of 1).
After normalizing to mean 0 / variance 1, two learned per-feature
parameters -- gamma (scale) and beta (shift) -- let the model undo the
normalization partially or entirely if that turns out to be useful;
they're initialized to 1 and 0 respectively, so an untrained LayerNorm
starts as a no-op beyond the normalization itself.

Uses the multi-head attention output from `attention/03_multi_head_attention.py`
as input, since that's exactly what feeds into a LayerNorm in a real
transformer block (see `03_full_transformer_block.py`).
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)

rng = np.random.default_rng(0)
# Stand-in for "some transformer sublayer's output" -- deliberately given
# a nonzero mean and a large scale (x10) to make LayerNorm's effect
# visible, rather than starting from already-small, near-normalized
# random noise.
X = rng.normal(loc=5.0, scale=10.0, size=(SEQ_LEN, D_MODEL))

print("=== Input internal state (BEFORE normalization) ===")
print(f"Shape: {X.shape}")
for pos, vec in enumerate(X):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): "
          f"mean={vec.mean():.3f}  std={vec.std():.3f}  values={np.round(vec, 2)}")


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return normalized * gamma + beta, mean, var


gamma = np.ones(D_MODEL)   # learned scale, starts as a no-op (1)
beta = np.zeros(D_MODEL)   # learned shift, starts as a no-op (0)

normalized, means, variances = layer_norm(X, gamma, beta)

print("\n=== Layer normalization internal state ===")
print(f"gamma (learned scale, at init): {gamma}")
print(f"beta (learned shift, at init):  {beta}")

print("\n=== Final output: per-position values after normalization ===")
for pos, vec in enumerate(normalized):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): "
          f"mean={vec.mean():.6f}  std={vec.std():.6f}  values={np.round(vec, 3)}")

print("\nNOTE: every position's mean is now ~0 and std is now ~1, "
      "REGARDLESS of how large or skewed its raw values were -- each "
      "position is normalized independently using only its own 8 "
      "features, never information from other positions or other "
      "sequences in a batch. With gamma=1, beta=0 (untrained "
      "initialization), the output IS the normalized value exactly; "
      "training would adjust gamma/beta to let the model rescale/shift "
      "specific features back up if that helps, without losing the "
      "stability normalization provides.")
