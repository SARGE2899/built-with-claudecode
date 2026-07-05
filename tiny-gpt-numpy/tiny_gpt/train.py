"""Stage 7: manual training loop.

Everything needed to train now exists: model_forward, cross_entropy,
model_backward all give us real gradients (stage 6, gradchecked). What's
left is bookkeeping:

  1. get_batch: slice random (input, target) windows out of the corpus.
     target is just input shifted one character to the right -- "predict
     the next character" is the entire training signal for a char-GPT.
  2. An optimizer: we implement Adam by hand (no library). Adam keeps a
     running mean (m) and running variance (v) of each gradient and uses
     them to adapt the effective step size per-parameter -- it converges
     much faster than plain SGD on this kind of ill-conditioned loss
     surface.
  3. The loop: forward -> loss -> backward -> optimizer step, repeat.

Since params/grads are nested dicts/lists (stage 6), the optimizer walks
that structure recursively rather than needing one flat vector.
"""

import numpy as np


def get_batch(data_ids, block_size, batch_size, rng):
    """Random (x, y) windows where y[i] = x[i] shifted right by one char."""
    max_start = len(data_ids) - block_size - 1
    ix = rng.integers(0, max_start, size=batch_size)
    x = np.stack([data_ids[i: i + block_size] for i in ix])
    y = np.stack([data_ids[i + 1: i + 1 + block_size] for i in ix])
    return x, y


def zeros_like_structure(x):
    """Build a nested dict/list/ndarray structure of zeros with the same
    shape as `x`. Non-array leaves (e.g. an `n_heads` int sitting next to
    Wq/Wk/... in the mha params dict) are dropped, matching how the grads
    dicts (stage 5/6) never contain those in the first place.
    """
    if isinstance(x, dict):
        return {k: zeros_like_structure(v) for k, v in x.items()
                 if isinstance(v, (dict, list, np.ndarray))}
    elif isinstance(x, list):
        return [zeros_like_structure(v) for v in x]
    elif isinstance(x, np.ndarray):
        return np.zeros_like(x)
    raise TypeError(f"unexpected leaf type {type(x)}")


class Adam:
    """Adam optimizer (Kingma & Ba, 2015), implemented from scratch.

    m = beta1*m + (1-beta1)*grad                 running mean of gradient
    v = beta2*v + (1-beta2)*grad^2                running mean of grad^2
    m_hat, v_hat = bias-corrected versions (m, v start at 0, biased early on)
    param -= lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.v = None
        self.t = 0
        self._params_template = params  # used to build m/v lazily from grads shape

    def step(self, params, grads):
        if self.m is None:
            self.m = zeros_like_structure(grads)
            self.v = zeros_like_structure(grads)
        self.t += 1
        self._step_recursive(params, grads, self.m, self.v)

    def _step_recursive(self, params, grads, m, v):
        if isinstance(params, dict):
            for k in grads:
                self._step_recursive(params[k], grads[k], m[k], v[k])
        elif isinstance(params, list):
            for p, g, mm, vv in zip(params, grads, m, v):
                self._step_recursive(p, g, mm, vv)
        elif isinstance(params, np.ndarray):
            m[:] = self.beta1 * m + (1 - self.beta1) * grads
            v[:] = self.beta2 * v + (1 - self.beta2) * (grads ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


if __name__ == "__main__":
    import os

    from model import (cross_entropy_backward, cross_entropy_forward,
                        init_model_params, model_backward, model_forward)
    from tokenizer import DATA_DIR, CharTokenizer

    with open(os.path.join(DATA_DIR, "corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data_ids = tok.encode(text)

    rng = np.random.default_rng(0)
    block_size, d_model, n_heads, d_ff, n_layers = 32, 32, 4, 128, 2
    params = init_model_params(tok.vocab_size, block_size, d_model, n_heads, d_ff, n_layers, seed=0)
    opt = Adam(params, lr=3e-3)

    batch_size = 16
    num_steps = 300
    print(f"corpus: {len(data_ids)} chars, vocab: {tok.vocab_size}, "
          f"model: d_model={d_model} n_heads={n_heads} n_layers={n_layers}")
    print(f"training for {num_steps} steps as a smoke test (real training run is stage 8)\n")

    losses = []
    for step in range(1, num_steps + 1):
        xb, yb = get_batch(data_ids, block_size, batch_size, rng)

        logits, cache = model_forward(xb, params)
        loss, ce_cache = cross_entropy_forward(logits, yb)
        dlogits = cross_entropy_backward(ce_cache)
        grads = model_backward(dlogits, cache, params)

        opt.step(params, grads)
        losses.append(loss)

        if step % 20 == 0 or step == 1:
            recent = np.mean(losses[-20:])
            print(f"step {step:4d}  loss {loss:.4f}  (avg last 20: {recent:.4f})")

    print(f"\nloss at step 1:   {losses[0]:.4f}")
    print(f"loss at step {num_steps}: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease -- training loop is broken"
    print("\nLoss decreased -- the training loop (batching + backward + Adam) works.")
