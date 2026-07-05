"""
Token Embeddings.

Concept: a token id (just an arbitrary integer, e.g. 'cat'=1) carries no
notion of meaning or similarity by itself -- id 1 isn't "closer" to id 2
than to id 4 in any meaningful sense. A token embedding table is a
learned lookup: one dense vector per vocabulary entry, so each token id
becomes a point in a continuous d_model-dimensional space where
(after training) semantically similar tokens end up near each other.
Looking up an id is just indexing a row of this matrix -- there is no
computation involved, which is why it's the very first layer of every
transformer-based LLM.

Uses the canonical vocabulary fixed in `tokenization/01_word_tokenization.py`
(the=0, cat=1, sat=2, on=3, mat=4) and the embedding dimension (d_model=8)
that every other script in this project reuses. The embedding table
itself is randomly initialized (as it would be before any training) --
we're demonstrating the lookup mechanism, not a trained result.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 8

SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]

rng = np.random.default_rng(0)
embedding_table = rng.normal(loc=0.0, scale=0.1, size=(VOCAB_SIZE, D_MODEL))

print("=== Token embedding internal state ===")
print(f"Embedding table shape: {embedding_table.shape} "
      f"(vocab_size={VOCAB_SIZE}, d_model={D_MODEL})")
print("Embedding table (one row per vocabulary token, randomly "
      "initialized -- this is what training would adjust):")
for word, idx in VOCAB.items():
    print(f"  row {idx} ({word!r}): {np.round(embedding_table[idx], 3)}")

print(f"\n=== Final output: embedding lookup for '{SENTENCE}' ===")
print(f"Token ids: {token_ids}")
embedded = embedding_table[token_ids]
print(f"Embedded sequence shape: {embedded.shape} (seq_len={len(token_ids)}, d_model={D_MODEL})")
for pos, (idx, vec) in enumerate(zip(token_ids, embedded)):
    print(f"  position {pos} ({ID_TO_WORD[idx]!r}, id={idx}): {np.round(vec, 3)}")

first_the = embedded[0]
second_the = embedded[4]
print(f"\nNOTE: position 0 ('the') and position 4 ('the') are IDENTICAL "
      f"vectors ({np.allclose(first_the, second_the)}) -- token embedding "
      "is a pure lookup by id, so every occurrence of the same token gets "
      "the exact same vector, with no awareness yet of WHERE in the "
      "sequence it appears. `03_combining_embeddings.py` shows how "
      "positional information gets added to break this symmetry.")
