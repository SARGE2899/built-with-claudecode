"""Stage 5c: multi-head attention, forward + backward, gradchecked.

Multi-head attention runs several independent attention "heads" in
parallel, each looking at a smaller slice (d_head = d_model / n_heads)
of the embedding, then concatenates their outputs and mixes them with
one more linear layer (Wo). This lets different heads specialize
(e.g. one head tracks "the previous vowel", another "the subject of
the sentence") instead of forcing one attention pattern to do everything.

The core scaled-dot-product-attention (SDPA) math is *exactly* the
same as stage 3/4's single-head attention -- we just add a leading
`n_heads` batch axis and rely on NumPy's matmul broadcasting rules
(`@` operates on the last two axes and broadcasts everything before
that), so no new math is needed for the core, only reshaping.

  x: (B, T, d_model)
  Q, K, V = x@Wq+bq, x@Wk+bk, x@Wv+bv        (B, T, d_model)
  split into heads: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
  attend per-head (identical math to stages 3/4, batched over B*n_heads)
  merge heads back: (B, n_heads, T, d_head) -> (B, T, d_model)
  out = merged @ Wo + bo
"""

import numpy as np

from attention import causal_mask, softmax


def init_mha_params(d_model, n_heads, seed=0):
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    rng = np.random.default_rng(seed)
    scale = 0.02
    return {
        "Wq": rng.normal(0, scale, size=(d_model, d_model)),
        "bq": np.zeros(d_model),
        "Wk": rng.normal(0, scale, size=(d_model, d_model)),
        "bk": np.zeros(d_model),
        "Wv": rng.normal(0, scale, size=(d_model, d_model)),
        "bv": np.zeros(d_model),
        "Wo": rng.normal(0, scale, size=(d_model, d_model)),
        "bo": np.zeros(d_model),
        "n_heads": n_heads,
    }


def split_heads(x, n_heads):
    """(B, T, d_model) -> (B, n_heads, T, d_head)"""
    B, T, d_model = x.shape
    d_head = d_model // n_heads
    x = x.reshape(B, T, n_heads, d_head)
    return x.transpose(0, 2, 1, 3)


def merge_heads(x):
    """(B, n_heads, T, d_head) -> (B, T, d_model)"""
    B, n_heads, T, d_head = x.shape
    x = x.transpose(0, 2, 1, 3)
    return x.reshape(B, T, n_heads * d_head)


def sdpa_forward(Q, K, V):
    """Scaled dot-product attention over the last two axes of Q,K,V.
    Q, K, V: (..., T, d) with identical leading '...' dims. Causal.
    """
    d_k = Q.shape[-1]
    T = Q.shape[-2]

    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)      # (..., T, T)
    mask = causal_mask(T)                                # (T, T)
    masked_scores = np.where(mask, scores, -np.inf)
    attn = softmax(masked_scores, axis=-1)
    out = attn @ V                                        # (..., T, d)

    cache = {"Q": Q, "K": K, "V": V, "attn": attn, "d_k": d_k}
    return out, cache


def sdpa_backward(dout, cache):
    Q, K, V, attn, d_k = cache["Q"], cache["K"], cache["V"], cache["attn"], cache["d_k"]

    dV = attn.swapaxes(-1, -2) @ dout
    dattn = dout @ V.swapaxes(-1, -2)

    s = np.sum(attn * dattn, axis=-1, keepdims=True)
    dscores = attn * (dattn - s) / np.sqrt(d_k)

    dQ = dscores @ K
    dK = dscores.swapaxes(-1, -2) @ Q

    return dQ, dK, dV


def mha_forward(x, params):
    n_heads = params["n_heads"]
    Wq, bq = params["Wq"], params["bq"]
    Wk, bk = params["Wk"], params["bk"]
    Wv, bv = params["Wv"], params["bv"]
    Wo, bo = params["Wo"], params["bo"]

    Q = x @ Wq + bq
    K = x @ Wk + bk
    V = x @ Wv + bv

    Qh, Kh, Vh = split_heads(Q, n_heads), split_heads(K, n_heads), split_heads(V, n_heads)
    out_heads, sdpa_cache = sdpa_forward(Qh, Kh, Vh)
    merged = merge_heads(out_heads)                       # (B, T, d_model)

    out = merged @ Wo + bo

    cache = {"x": x, "merged": merged, "sdpa_cache": sdpa_cache,
              "n_heads": n_heads, "params": params}
    return out, cache


def mha_backward(dout, cache):
    x, merged, sdpa_cache, n_heads, params = (
        cache["x"], cache["merged"], cache["sdpa_cache"], cache["n_heads"], cache["params"])
    Wq, Wk, Wv, Wo = params["Wq"], params["Wk"], params["Wv"], params["Wo"]

    sum_axes = (0, 1)
    dWo = np.einsum("bti,btj->ij", merged, dout)
    dbo = np.sum(dout, axis=sum_axes)
    dmerged = dout @ Wo.T

    dout_heads = split_heads(dmerged, n_heads)
    dQh, dKh, dVh = sdpa_backward(dout_heads, sdpa_cache)

    dQ = merge_heads(dQh)
    dK = merge_heads(dKh)
    dV = merge_heads(dVh)

    dWq = np.einsum("bti,btj->ij", x, dQ)
    dbq = np.sum(dQ, axis=sum_axes)
    dWk = np.einsum("bti,btj->ij", x, dK)
    dbk = np.sum(dK, axis=sum_axes)
    dWv = np.einsum("bti,btj->ij", x, dV)
    dbv = np.sum(dV, axis=sum_axes)

    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    grads = {"Wq": dWq, "bq": dbq, "Wk": dWk, "bk": dbk,
              "Wv": dWv, "bv": dbv, "Wo": dWo, "bo": dbo}
    return dx, grads


if __name__ == "__main__":
    import sys

    from gradcheck import check_gradient, numerical_gradient

    rng = np.random.default_rng(0)
    B, T, d_model, n_heads = 2, 6, 8, 2
    x = rng.normal(size=(B, T, d_model))
    params = init_mha_params(d_model, n_heads)
    for k in ("bq", "bk", "bv", "bo"):
        params[k] = rng.normal(0, 0.02, size=d_model)

    out, cache = mha_forward(x, params)
    print(f"x shape: {x.shape} -> out shape: {out.shape}  ({n_heads} heads x {d_model // n_heads} d_head)")

    print("\n--- gradient check: multi-head attention backward ---")
    dout_seed = rng.normal(size=out.shape)

    def loss_fn(x_, params_):
        out_, _ = mha_forward(x_, params_)
        return np.sum(out_ * dout_seed)

    dx, grads = mha_backward(dout_seed, cache)

    ok = True

    x_work = x.copy()
    num_dx = numerical_gradient(lambda: loss_fn(x_work, params), x_work)
    ok &= check_gradient("dx", dx, num_dx)

    for key in ("Wq", "bq", "Wk", "bk", "Wv", "bv", "Wo", "bo"):
        work = params[key].copy()
        params_work = dict(params, **{key: work})
        num = numerical_gradient(lambda: loss_fn(x, params_work), work)
        ok &= check_gradient(f"d{key}", grads[key], num)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- multi-head attention backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED.")
        sys.exit(1)
