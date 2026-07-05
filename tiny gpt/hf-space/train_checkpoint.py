"""Trains tiny_gpt on the sample corpus and saves a self-contained checkpoint
(params + tokenizer vocab + hyperparams) as checkpoint.pkl, so the Space can
load a ready-to-generate model at startup instead of training on boot.

Run once locally to (re)produce checkpoint.pkl:
    python train_checkpoint.py
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny_gpt"))

from model import (cross_entropy_backward, cross_entropy_forward,
                    init_model_params, model_backward, model_forward)
from tokenizer import CharTokenizer
from train import Adam, get_batch

CORPUS = """the quick brown fox jumps over the lazy dog.
the dog barks at the fox. the fox runs away.
the sun is bright and the sky is blue.
a cat sat on the mat and looked at the sun.
the quick cat chased the lazy dog around the yard.
the dog and the cat became good friends in the end.
"""

HYPERPARAMS = dict(block_size=32, d_model=32, n_heads=4, d_ff=128, n_layers=2)
NUM_STEPS = 2000
BATCH_SIZE = 16
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pkl")


def main():
    tok = CharTokenizer(CORPUS)
    data_ids = tok.encode(CORPUS)

    rng = np.random.default_rng(0)
    params = init_model_params(tok.vocab_size, **HYPERPARAMS, seed=0)
    opt = Adam(params, lr=3e-3)

    print(f"corpus: {len(data_ids)} chars, vocab: {tok.vocab_size}")
    for step in range(1, NUM_STEPS + 1):
        xb, yb = get_batch(data_ids, HYPERPARAMS["block_size"], BATCH_SIZE, rng)
        logits, cache = model_forward(xb, params)
        loss, ce_cache = cross_entropy_forward(logits, yb)
        dlogits = cross_entropy_backward(ce_cache)
        grads = model_backward(dlogits, cache, params)
        opt.step(params, grads)

        if step % 200 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss:.4f}")

    checkpoint = {
        "params": params,
        "stoi": tok.stoi,
        "itos": tok.itos,
        "vocab_size": tok.vocab_size,
        "hyperparams": HYPERPARAMS,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nsaved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
