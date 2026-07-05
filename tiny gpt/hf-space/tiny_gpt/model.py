"""Stage 6: full model -- embeddings -> N transformer blocks -> final LN ->
output projection -> softmax -> cross-entropy loss. Forward + backward,
gradchecked end-to-end.

This closes the loop left open in stage 2: token/positional embedding
lookup is just indexing (tok_emb[ids]), so its backward is "scatter-add
the incoming gradient into the rows that were looked up" -- if a
character appears at 5 positions across the batch, its embedding row's
gradient is the sum of the 5 gradients flowing back to those positions.

  x      = tok_emb[ids] + pos_emb[:T]
  x      = block_1(x); block_2(x); ...; block_L(x)      (stage 5, stacked)
  x      = LN_final(x)
  logits = x @ Wout + bout                                (B, T, vocab_size)
  probs  = softmax(logits, axis=-1)
  loss   = mean_{b,t}( -log(probs[b, t, targets[b, t]]) )

Cross-entropy + softmax backward has the famously clean closed form
dlogits = probs - one_hot(targets) (scaled by 1/(B*T) for the mean) --
we still gradcheck it like everything else rather than trust that from
memory.
"""

import numpy as np

from attention import softmax
from block import block_backward, block_forward, init_block_params
from layernorm import layernorm_backward, layernorm_forward


def init_model_params(vocab_size, block_size, d_model, n_heads, d_ff, n_layers, seed=0):
    rng = np.random.default_rng(seed)
    scale = 0.02
    return {
        "tok_emb": rng.normal(0, scale, size=(vocab_size, d_model)),
        "pos_emb": rng.normal(0, scale, size=(block_size, d_model)),
        "blocks": [init_block_params(d_model, n_heads, d_ff, seed=seed + i)
                   for i in range(n_layers)],
        "lnf_gamma": np.ones(d_model),
        "lnf_beta": np.zeros(d_model),
        "Wout": rng.normal(0, scale, size=(d_model, vocab_size)),
        "bout": np.zeros(vocab_size),
    }


def model_forward(ids, params):
    B, T = ids.shape
    tok_emb, pos_emb = params["tok_emb"], params["pos_emb"]

    x = tok_emb[ids] + pos_emb[:T][None, :]

    block_caches = []
    for blk_params in params["blocks"]:
        x, blk_cache = block_forward(x, blk_params)
        block_caches.append(blk_cache)

    x_before_lnf = x
    x, lnf_cache = layernorm_forward(x, params["lnf_gamma"], params["lnf_beta"])

    logits = x @ params["Wout"] + params["bout"]

    cache = {
        "ids": ids, "T": T, "x_lnf_out": x,
        "block_caches": block_caches, "lnf_cache": lnf_cache,
    }
    return logits, cache


def cross_entropy_forward(logits, targets):
    """logits: (B, T, V), targets: (B, T) int. Mean NLL over all B*T tokens."""
    B, T, V = logits.shape
    probs = softmax(logits, axis=-1)
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    correct_probs = probs[b_idx, t_idx, targets]   # (B, T)
    loss = np.mean(-np.log(correct_probs))
    cache = {"probs": probs, "targets": targets}
    return loss, cache


def cross_entropy_backward(cache):
    probs, targets = cache["probs"], cache["targets"]
    B, T, V = probs.shape
    dlogits = probs.copy()
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    dlogits[b_idx, t_idx, targets] -= 1.0
    dlogits /= (B * T)
    return dlogits


def model_backward(dlogits, cache, params):
    ids, T = cache["ids"], cache["T"]
    x_lnf_out, block_caches, lnf_cache = cache["x_lnf_out"], cache["block_caches"], cache["lnf_cache"]
    B = ids.shape[0]

    dWout = np.einsum("btd,btv->dv", x_lnf_out, dlogits)
    dbout = np.sum(dlogits, axis=(0, 1))
    dx = dlogits @ params["Wout"].T

    dx, dlnf_gamma, dlnf_beta = layernorm_backward(dx, lnf_cache)

    block_grads = []
    for blk_params, blk_cache in zip(reversed(params["blocks"]), reversed(block_caches)):
        dx, grads = block_backward(dx, blk_cache)
        block_grads.append(grads)
    block_grads.reverse()

    # embedding backward: x = tok_emb[ids] + pos_emb[:T], scatter-add since
    # the same row of tok_emb may have been looked up at multiple (b, t).
    dtok_emb = np.zeros_like(params["tok_emb"])
    np.add.at(dtok_emb, ids, dx)
    dpos_emb = np.zeros_like(params["pos_emb"])
    dpos_emb[:T] = dx.sum(axis=0)

    grads = {
        "tok_emb": dtok_emb, "pos_emb": dpos_emb,
        "blocks": block_grads,
        "lnf_gamma": dlnf_gamma, "lnf_beta": dlnf_beta,
        "Wout": dWout, "bout": dbout,
    }
    return grads


if __name__ == "__main__":
    import sys

    from gradcheck import check_gradient, numerical_gradient, walk_params

    rng = np.random.default_rng(0)
    vocab_size, block_size, d_model, n_heads, d_ff, n_layers = 10, 8, 8, 2, 16, 2
    params = init_model_params(vocab_size, block_size, d_model, n_heads, d_ff, n_layers)
    # nonzero biases everywhere so gradcheck exercises every parameter
    for blk in params["blocks"]:
        for k in ("bq", "bk", "bv", "bo"):
            blk["mha"][k] = rng.normal(0, 0.02, size=d_model)
        blk["ffn"]["b1"] = rng.normal(0, 0.02, size=d_ff)
        blk["ffn"]["b2"] = rng.normal(0, 0.02, size=d_model)
    params["bout"] = rng.normal(0, 0.02, size=vocab_size)

    B, T = 3, 6
    ids = rng.integers(0, vocab_size, size=(B, T))
    targets = rng.integers(0, vocab_size, size=(B, T))

    logits, cache = model_forward(ids, params)
    loss, ce_cache = cross_entropy_forward(logits, targets)
    print(f"ids shape: {ids.shape} -> logits shape: {logits.shape}")
    print(f"loss: {loss:.4f}  (random-init sanity check, ~ln(vocab_size)={np.log(vocab_size):.4f})")

    dlogits = cross_entropy_backward(ce_cache)
    grads = model_backward(dlogits, cache, params)

    print("\n--- gradient check: full model + cross-entropy backward ---")

    def loss_fn():
        logits_, _ = model_forward(ids, params)
        loss_, _ = cross_entropy_forward(logits_, targets)
        return loss_

    ok = True
    for path, p_arr, g_arr in walk_params(params, grads):
        num = numerical_gradient(loss_fn, p_arr)
        ok &= check_gradient(path, g_arr, num)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- full model + loss backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED -- do not proceed to stage 7.")
        sys.exit(1)
