"""
Positional Encoding.

Concept: self-attention (see `attention/01_scaled_dot_product_attention.py`)
computes a weighted average over ALL positions at once, with no built-in
notion of order -- shuffle the input tokens, and unmodified self-attention
would shuffle its output the exact same way. Positional encoding fixes
this by adding position-dependent information to each token's vector
before it ever reaches attention, so position 0 and position 4 become
distinguishable even when they hold the same token ('the').

Two schemes are compared here:
  1. SINUSOIDAL (the original Transformer's choice) -- a fixed formula,
     no learned parameters. Each position gets a unique pattern of sine/
     cosine values at geometrically increasing wavelengths across the
     d_model dimensions, so nearby positions produce similar (but
     distinguishable) patterns.
  2. LEARNED positional embeddings -- structurally identical to a token
     embedding table (`01_token_embeddings.py`), but indexed by POSITION
     instead of token id, with values adjusted during training rather
     than fixed by a formula. Shown here at random initialization, since
     no training happens in this project.

Uses the same d_model=8 and the canonical sentence's sequence length
(6 positions) as every other script in this project.
"""

import numpy as np

D_MODEL = 8
SEQ_LEN = 6  # "the cat sat on the mat" -> 6 tokens


def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None]              # (seq_len, 1)
    dim = np.arange(d_model)[None, :]                    # (1, d_model)
    angle_rates = 1.0 / (10000 ** (2 * (dim // 2) / d_model))
    angles = position * angle_rates                      # (seq_len, d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])   # even dims: sine
    pe[:, 1::2] = np.cos(angles[:, 1::2])   # odd dims: cosine
    return pe


print("=== Sinusoidal positional encoding internal state ===")
sinusoidal_pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
print(f"Shape: {sinusoidal_pe.shape} (seq_len={SEQ_LEN}, d_model={D_MODEL})")
for pos in range(SEQ_LEN):
    print(f"  position {pos}: {np.round(sinusoidal_pe[pos], 4)}")

print("\nNOTE: dimension 0 (sin) cycles fastest across positions; higher "
      "dimension pairs use longer wavelengths (the 1/10000^(2i/d_model) "
      "term), so nearby positions look similar in the fast-changing early "
      "dimensions but only diverge in the slow-changing later ones -- "
      "this geometric spread is what gives the model a notion of relative "
      "distance between positions, not just an arbitrary unique ID each.")

rng = np.random.default_rng(0)
learned_pe = rng.normal(loc=0.0, scale=0.1, size=(SEQ_LEN, D_MODEL))

print("\n=== Learned positional embedding internal state (untrained) ===")
print(f"Shape: {learned_pe.shape} (identical shape to the sinusoidal table above)")
for pos in range(SEQ_LEN):
    print(f"  position {pos}: {np.round(learned_pe[pos], 4)}")
print("\nNOTE: structurally this is just an embedding table indexed by "
      "position instead of token id -- at initialization it's random "
      "noise with no relationship between neighboring positions at all. "
      "Only training would shape it into something with real structure. "
      "Its practical downside vs. sinusoidal: it has no defined values "
      "beyond the max sequence length seen during training, whereas the "
      "sinusoidal formula can be evaluated at any position, even ones "
      "longer than anything seen in training.")

print("\n=== Final output: comparing both schemes at position 0 vs 4 ===")
print("(positions 0 and 4 both hold the token 'the' in the canonical "
      "sentence -- this is exactly the pair whose token embeddings are "
      "identical in `01_token_embeddings.py`)")
print(f"Sinusoidal PE[0] vs PE[4] equal? "
      f"{np.allclose(sinusoidal_pe[0], sinusoidal_pe[4])}")
print(f"Learned PE[0] vs PE[4] equal?    "
      f"{np.allclose(learned_pe[0], learned_pe[4])}")
print("Both schemes assign different vectors to positions 0 and 4, which "
      "is the entire point: adding either of these to the (identical) "
      "token embedding for 'the' at each position will make the two "
      "occurrences distinguishable to the model.")
