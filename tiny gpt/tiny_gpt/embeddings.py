"""Stage 2: token + positional embeddings (forward only).

Every model input is a sequence of integer token ids, shape (B, T).
We turn each id into a vector two ways and add them:

  tok_emb : (vocab_size, d_model)  -- a learned row per character
  pos_emb : (block_size, d_model)  -- a learned row per *position* 0..T-1

x[b, t, :] = tok_emb[ids[b, t]] + pos_emb[t]

The token embedding tells the model "which character is this."
The positional embedding tells it "where in the sequence am I" --
without it, attention (stage 3) has no notion of order at all, since
attention itself treats the sequence as an unordered set of vectors.

No backward pass here yet: gradient for an embedding lookup is just
"scatter-add the incoming gradient into the row(s) that were looked
up," which we'll implement once we do full backprop in stage 6.
"""

import os

import numpy as np

from tokenizer import DATA_DIR, CharTokenizer


def init_embeddings(vocab_size, block_size, d_model, seed=0):
    rng = np.random.default_rng(seed)
    tok_emb = rng.normal(0, 0.02, size=(vocab_size, d_model))
    pos_emb = rng.normal(0, 0.02, size=(block_size, d_model))
    return tok_emb, pos_emb


def embed(ids, tok_emb, pos_emb):
    """ids: (B, T) int array -> x: (B, T, d_model)"""
    B, T = ids.shape
    tok = tok_emb[ids]           # (B, T, d_model), fancy indexing = lookup
    pos = pos_emb[:T][None, :]   # (1, T, d_model), broadcasts over batch
    return tok + pos


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)

    d_model = 8
    block_size = 16
    tok_emb, pos_emb = init_embeddings(tok.vocab_size, block_size, d_model)

    batch_strs = ["the quick br", "og barks at "]
    ids = np.stack([tok.encode(s) for s in batch_strs])  # (B=2, T=12)
    print(f"ids shape: {ids.shape}")

    x = embed(ids, tok_emb, pos_emb)
    print(f"x shape: {x.shape}  (B, T, d_model)")

    # sanity check 1: same char at different positions -> different vectors
    # 't' appears at position 0 in both strings (index of 't' = stoi['t'])
    t_id = tok.stoi["t"]
    assert ids[0, 0] == t_id
    print(f"\n't' at position 0, seq 0, vector[:4]: {x[0, 0, :4]}")

    # sanity check 2: subtracting the positional term recovers pure token embedding
    recovered = x[0, 0] - pos_emb[0]
    assert np.allclose(recovered, tok_emb[t_id])
    print("token/position decomposition check OK")

    # sanity check 3: two different sequences sharing a prefix char at the same
    # position get the same positional contribution but different token content
    diff = x[0, 0] - x[1, 0]
    expected_diff = tok_emb[ids[0, 0]] - tok_emb[ids[1, 0]]
    assert np.allclose(diff, expected_diff)
    print("position-independence-of-token-diff check OK")
