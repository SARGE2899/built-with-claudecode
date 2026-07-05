"""Stage 5d: full transformer block = LN + MHA + residual + LN + FFN + residual.

This is the "pre-LN" architecture (GPT-2 style): LayerNorm is applied
*before* each sublayer, not after, and the sublayer's output is added
back to its input via a residual connection:

  x1 = x  + MHA(LN1(x))
  x2 = x1 + FFN(LN2(x1))

Residual connections are why deep transformers are trainable at all:
in the backward pass, d(x2)/d(x1) contains an identity term (from the
"+"), so gradients have a direct, unattenuated path all the way back to
the input regardless of how many blocks are stacked -- they don't have
to survive being multiplied through every sublayer's Jacobian.

Backward here is just the reverse topological order of the forward
graph: unwind the second residual+FFN+LN, then the first
residual+MHA+LN, adding the two paths at each "+".
"""

import numpy as np

from feedforward import ffn_backward, ffn_forward, init_ffn_params
from layernorm import layernorm_backward, layernorm_forward
from mha import init_mha_params, mha_backward, mha_forward


def init_block_params(d_model, n_heads, d_ff, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "ln1_gamma": np.ones(d_model), "ln1_beta": np.zeros(d_model),
        "ln2_gamma": np.ones(d_model), "ln2_beta": np.zeros(d_model),
        "mha": init_mha_params(d_model, n_heads, seed=seed),
        "ffn": init_ffn_params(d_model, d_ff, seed=seed + 1),
    }


def block_forward(x, params):
    ln1_out, ln1_cache = layernorm_forward(x, params["ln1_gamma"], params["ln1_beta"])
    attn_out, attn_cache = mha_forward(ln1_out, params["mha"])
    x1 = x + attn_out

    ln2_out, ln2_cache = layernorm_forward(x1, params["ln2_gamma"], params["ln2_beta"])
    ffn_out, ffn_cache = ffn_forward(ln2_out, params["ffn"])
    x2 = x1 + ffn_out

    cache = {"ln1_cache": ln1_cache, "attn_cache": attn_cache,
              "ln2_cache": ln2_cache, "ffn_cache": ffn_cache}
    return x2, cache


def block_backward(dx2, cache):
    ln1_cache, attn_cache = cache["ln1_cache"], cache["attn_cache"]
    ln2_cache, ffn_cache = cache["ln2_cache"], cache["ffn_cache"]

    # x2 = x1 + ffn_out  -> gradient splits identically to both branches
    dx1_res = dx2
    dffn_out = dx2
    dln2_out, dW1, db1, dW2, db2 = ffn_backward(dffn_out, ffn_cache)
    dx1_ln, dln2_gamma, dln2_beta = layernorm_backward(dln2_out, ln2_cache)
    dx1 = dx1_res + dx1_ln

    # x1 = x + attn_out
    dx_res = dx1
    dattn_out = dx1
    dln1_out, mha_grads = mha_backward(dattn_out, attn_cache)
    dx_ln, dln1_gamma, dln1_beta = layernorm_backward(dln1_out, ln1_cache)
    dx = dx_res + dx_ln

    grads = {
        "ln1_gamma": dln1_gamma, "ln1_beta": dln1_beta,
        "ln2_gamma": dln2_gamma, "ln2_beta": dln2_beta,
        "mha": mha_grads,
        "ffn": {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2},
    }
    return dx, grads


if __name__ == "__main__":
    import sys

    from gradcheck import check_gradient, numerical_gradient

    rng = np.random.default_rng(0)
    B, T, d_model, n_heads, d_ff = 2, 6, 8, 2, 16
    x = rng.normal(size=(B, T, d_model))
    params = init_block_params(d_model, n_heads, d_ff)
    # nonzero biases everywhere so gradcheck exercises every parameter
    for k in ("bq", "bk", "bv", "bo"):
        params["mha"][k] = rng.normal(0, 0.02, size=d_model)
    params["ffn"]["b1"] = rng.normal(0, 0.02, size=d_ff)
    params["ffn"]["b2"] = rng.normal(0, 0.02, size=d_model)

    out, cache = block_forward(x, params)
    print(f"x shape: {x.shape} -> block out shape: {out.shape}")

    print("\n--- gradient check: full transformer block backward ---")
    dout_seed = rng.normal(size=out.shape)

    def loss_fn(x_, params_):
        out_, _ = block_forward(x_, params_)
        return np.sum(out_ * dout_seed)

    dx, grads = block_backward(dout_seed, cache)

    ok = True

    x_work = x.copy()
    num_dx = numerical_gradient(lambda: loss_fn(x_work, params), x_work)
    ok &= check_gradient("dx", dx, num_dx)

    # top-level LN params
    for key in ("ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"):
        work = params[key].copy()
        params_work = dict(params, **{key: work})
        num = numerical_gradient(lambda: loss_fn(x, params_work), work)
        ok &= check_gradient(key, grads[key], num)

    # nested mha params
    for key in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
        work = params["mha"][key].copy()
        mha_work = dict(params["mha"], **{key: work})
        params_work = dict(params, mha=mha_work)
        num = numerical_gradient(lambda: loss_fn(x, params_work), work)
        ok &= check_gradient(f"mha.{key}", grads["mha"][key], num)

    # nested ffn params
    for key in ("W1", "b1", "W2", "b2"):
        work = params["ffn"][key].copy()
        ffn_work = dict(params["ffn"], **{key: work})
        params_work = dict(params, ffn=ffn_work)
        num = numerical_gradient(lambda: loss_fn(x, params_work), work)
        ok &= check_gradient(f"ffn.{key}", grads["ffn"][key], num)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- full transformer block backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED -- do not proceed to stage 6.")
        sys.exit(1)
