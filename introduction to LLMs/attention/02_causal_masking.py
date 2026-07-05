"""
Causal (Autoregressive) Masking.

Concept: `01_scaled_dot_product_attention.py` let every position attend
to every other position, including ones later in the sequence. An
autoregressive LLM generates text left-to-right, one token at a time,
and must be trained under the exact same constraint it will face at
generation time: when predicting position i, it can only see positions
0..i, never i+1 onward (which don't exist yet at generation time, and
would leak the "answer" during training otherwise). The fix: before the
softmax step, set every score at position (i, j) where j > i to -infinity,
so softmax assigns it exactly 0 weight -- the position simply cannot
contribute to the output, no matter what Q/K values say.

Same setup as `01_scaled_dot_product_attention.py` (same X, same
Q/K/V projections) so the two scripts' attention weights can be
compared directly, before vs. after masking.
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
Q, K, V = X @ W_q, X @ W_k, X @ W_v

d_k = K.shape[-1]
scaled_scores = (Q @ K.T) / np.sqrt(d_k)

print("=== Building the causal mask ===")
# Upper triangle (excluding diagonal) = future positions = masked out.
causal_mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)
print("Mask (True = position masked out, i.e. 'future, cannot attend'):")
print(causal_mask)

masked_scores = np.where(causal_mask, -np.inf, scaled_scores)
print("\nScores after masking (-inf where masked; softmax will zero these):")
with np.printoptions(precision=2, suppress=True):
    print(masked_scores)

masked_attention_weights = softmax(masked_scores, axis=-1)
unmasked_attention_weights = softmax(scaled_scores, axis=-1)

print("\n=== Final output: attention weights, unmasked vs. causally masked ===")
header = "        " + "  ".join(f"{ID_TO_WORD[t]:>5s}" for t in token_ids)
print(f"\nUNMASKED (from 01_scaled_dot_product_attention.py):\n{header}")
for pos, row in enumerate(unmasked_attention_weights):
    print(f"  {ID_TO_WORD[token_ids[pos]]:>5s}: {np.round(row, 3)}")

print(f"\nCAUSALLY MASKED:\n{header}")
for pos, row in enumerate(masked_attention_weights):
    print(f"  {ID_TO_WORD[token_ids[pos]]:>5s}: {np.round(row, 3)}  (sum={row.sum():.3f})")

print("\nNOTE: the masked weight matrix is lower-triangular -- position 0 "
      "('the') puts 100% of its weight on itself (it's the only position "
      "it's allowed to see), while position 5 ('mat', the last token) can "
      "still attend across the whole sequence, since everything before it "
      "is fair game. This asymmetry -- early positions see little context, "
      "late positions see everything before them -- is exactly the "
      "left-to-right structure an autoregressive LLM both trains and "
      "generates under.")
