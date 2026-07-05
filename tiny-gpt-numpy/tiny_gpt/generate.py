"""Stage 8: train to convergence on the tiny corpus, then generate text.

Generation is autoregressive: feed the model the last `block_size`
characters, take the logits for the *final* position (its prediction for
"what comes next"), turn them into a probability distribution, sample one
character from it, append, and repeat. Each new character becomes part of
the context for the next step -- there is no separate "generation mode",
it's the exact same model_forward used in training.

temperature reshapes the distribution before sampling:
  T < 1: sharpen it (favor the model's top choices, more deterministic)
  T = 1: sample the model's distribution as-is
  T > 1: flatten it (more random/diverse, can go incoherent)
"""

import os

import numpy as np

from attention import softmax
from model import (cross_entropy_backward, cross_entropy_forward,
                    init_model_params, model_backward, model_forward)
from tokenizer import DATA_DIR, CharTokenizer
from train import Adam, get_batch


def generate(params, tok, prompt, max_new_tokens, block_size, temperature=1.0, rng=None):
    rng = rng or np.random.default_rng()
    ids = list(tok.encode(prompt))

    for _ in range(max_new_tokens):
        context = np.array(ids[-block_size:])[None, :]          # (1, T)
        logits, _ = model_forward(context, params)
        last_logits = logits[0, -1, :] / temperature              # (vocab_size,)
        probs = softmax(last_logits, axis=-1)
        next_id = rng.choice(len(probs), p=probs)
        ids.append(next_id)

    return tok.decode(ids)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data_ids = tok.encode(text)

    rng = np.random.default_rng(0)
    block_size, d_model, n_heads, d_ff, n_layers = 32, 32, 4, 128, 2
    params = init_model_params(tok.vocab_size, block_size, d_model, n_heads, d_ff, n_layers, seed=0)
    opt = Adam(params, lr=3e-3)

    batch_size = 16
    num_steps = 2000
    print(f"corpus: {len(data_ids)} chars, vocab: {tok.vocab_size}")
    print(f"training for {num_steps} steps...\n")

    for step in range(1, num_steps + 1):
        xb, yb = get_batch(data_ids, block_size, batch_size, rng)
        logits, cache = model_forward(xb, params)
        loss, ce_cache = cross_entropy_forward(logits, yb)
        dlogits = cross_entropy_backward(ce_cache)
        grads = model_backward(dlogits, cache, params)
        opt.step(params, grads)

        if step % 200 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss:.4f}")

    print("\n--- generation ---")
    gen_rng = np.random.default_rng(123)
    for prompt in ["the ", "the dog ", "a cat "]:
        for temp in (0.5, 1.0):
            out = generate(params, tok, prompt, max_new_tokens=60,
                             block_size=block_size, temperature=temp, rng=gen_rng)
            print(f"[temp={temp}] {out!r}")
        print()
