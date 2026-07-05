"""
Position-wise Feedforward Network + Residual Connections.

Concept, part 1 (feedforward): attention mixes information ACROSS
positions; the feedforward block is the complementary step that
processes EACH position independently through a small 2-layer MLP
(Linear -> nonlinearity -> Linear), typically expanding to a wider
hidden dimension (here 4x d_model) before projecting back down. Every
position is passed through the exact same weights (hence "position-
wise") -- there's no mixing between positions here at all, that's
entirely attention's job.

Concept, part 2 (residuals): rather than a sublayer (attention or
feedforward) replacing its input outright, its output is ADDED back to
its own input: `output = X + Sublayer(X)`. This gives every sublayer an
easy "do nothing" starting point (if Sublayer(X) is near zero -- as it
tends to be early in training with small weights -- the residual just
passes X through unchanged), and critically, it gives gradients a direct
path (the `+ X` term) straight back through many stacked layers during
backpropagation, without having to flow through every intermediate
sublayer's transformation.

Uses the same d_model=8 and canonical sentence as every other script.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
D_FF = D_MODEL * 4  # standard convention: feedforward hidden dim = 4x d_model
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)

rng = np.random.default_rng(0)
# Stand-in for "the output of a previous sublayer" (e.g. post-attention).
X = rng.normal(scale=1.0, size=(SEQ_LEN, D_MODEL))

W1 = rng.normal(scale=0.3, size=(D_MODEL, D_FF))
b1 = np.zeros(D_FF)
W2 = rng.normal(scale=0.3, size=(D_FF, D_MODEL))
b2 = np.zeros(D_MODEL)


def relu(x):
    return np.maximum(0, x)


print("=== Feedforward internal state ===")
print(f"Input X shape: {X.shape}")
print(f"W1 shape: {W1.shape} (d_model={D_MODEL} -> d_ff={D_FF}, a 4x expansion)")
print(f"W2 shape: {W2.shape} (d_ff={D_FF} -> back down to d_model={D_MODEL})")

hidden = relu(X @ W1 + b1)
print(f"\nHidden activation shape: {hidden.shape}")
print(f"Fraction of hidden units that are exactly 0 (ReLU killed them): "
      f"{(hidden == 0).mean():.2%}")

ffn_out = hidden @ W2 + b2
print(f"Feedforward output shape: {ffn_out.shape} (back to d_model, matching input X)")

print("\n=== Residual connection internal state ===")
residual_out = X + ffn_out
print("Per-position comparison of ||X|| vs ||Sublayer(X)|| vs ||output||:")
for pos in range(SEQ_LEN):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): "
          f"||X||={np.linalg.norm(X[pos]):.3f}  "
          f"||FFN(X)||={np.linalg.norm(ffn_out[pos]):.3f}  "
          f"||X + FFN(X)||={np.linalg.norm(residual_out[pos]):.3f}")

print("\n=== Final output: X + FFN(X), per position ===")
for pos, vec in enumerate(residual_out):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): {np.round(vec, 3)}")

print("\nNOTE: with W1/W2 randomly initialized at a modest scale (0.3), "
      "FFN(X) is roughly the same order of magnitude as X itself here -- "
      "in a real, freshly-initialized network weights are often scaled "
      "down further specifically so that FFN(X) starts small relative to "
      "X, making the residual path dominate early in training (the "
      "network starts close to 'identity plus a small correction' at "
      "every layer, which is much easier to optimize than starting from "
      "an arbitrary transformation at every layer).")
