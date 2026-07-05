"""
Combining Token and Positional Embeddings.

Concept: the actual input to a transformer's first layer is neither
embedding alone -- it's their ELEMENT-WISE SUM: for each position,
token_embedding[token_id at that position] + positional_encoding[that
position]. This single combined vector is what carries both "which
token is this" and "where in the sequence is it" into every downstream
layer; attention and everything after it only ever sees this sum, never
the two components separately.

Combines exactly what `01_token_embeddings.py` and
`02_positional_encoding.py` computed independently, using the same
canonical vocabulary, d_model=8, and the same sinusoidal formula.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)

rng = np.random.default_rng(0)
token_embedding_table = rng.normal(loc=0.0, scale=0.1, size=(len(VOCAB), D_MODEL))
token_embeddings = token_embedding_table[token_ids]


def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None]
    dim = np.arange(d_model)[None, :]
    angle_rates = 1.0 / (10000 ** (2 * (dim // 2) / d_model))
    angles = position * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


positional_encoding = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)

print("=== Internal state: the two components being combined ===")
print(f"Token embeddings shape: {token_embeddings.shape}")
print(f"Positional encoding shape: {positional_encoding.shape}")

combined = token_embeddings + positional_encoding

print("\n=== Final output: combined input embeddings (token + position) ===")
for pos, (idx, tok_vec, pos_vec, comb_vec) in enumerate(
    zip(token_ids, token_embeddings, positional_encoding, combined)
):
    print(f"\nposition {pos} ({ID_TO_WORD[idx]!r}, id={idx}):")
    print(f"  token embedding:     {np.round(tok_vec, 3)}")
    print(f"  positional encoding: {np.round(pos_vec, 3)}")
    print(f"  combined (sum):      {np.round(comb_vec, 3)}")

print("\n=== Why this matters: the two 'the' tokens (positions 0 and 4) ===")
print(f"Token embeddings identical?      "
      f"{np.allclose(token_embeddings[0], token_embeddings[4])}")
print(f"Positional encodings identical?  "
      f"{np.allclose(positional_encoding[0], positional_encoding[4])}")
print(f"Combined vectors identical?      "
      f"{np.allclose(combined[0], combined[4])}")
print("\nBefore this step, both occurrences of 'the' were represented by "
      "the exact same vector -- a model operating on token embeddings "
      "alone genuinely cannot tell them apart. After adding position-"
      "specific information, they become distinct vectors while still "
      "sharing whatever 'the'-ness the token embedding itself encodes -- "
      "which is exactly the combination every later transformer layer "
      "(starting with self-attention) actually operates on.")
