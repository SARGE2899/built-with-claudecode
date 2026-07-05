"""Stage 1: character-level tokenizer.

The simplest possible tokenizer: the vocabulary is just every unique
character that appears in the training text. No merges, no subwords.
"""

import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"corpus length: {len(text)} chars")
    print(f"vocab size: {tok.vocab_size}")
    print(f"vocab: {sorted(tok.stoi.keys())!r}")

    sample = text[:40]
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"\nsample: {sample!r}")
    print(f"encoded: {ids}")
    print(f"decoded: {back!r}")
    assert back == sample, "round-trip failed!"
    print("\nround-trip OK")
