"""Stage 3: single-head causal self-attention (forward only).

For each position t, attention builds a weighted average of the *value*
vectors of all positions <= t (causal: a GPT must not look at the future).
The weights come from how well position t's "query" matches every other
position's "key".

Given x: (B, T, d_model)

  Q = x @ Wq              (B, T, d_k)
  K = x @ Wk              (B, T, d_k)
  V = x @ Wv              (B, T, d_v)

  scores = Q @ K^T / sqrt(d_k)     (B, T, T)   scores[b,i,j] = query_i . key_j
  scores = mask future (j > i) to -inf         so softmax gives them weight 0
  attn   = softmax(scores, axis=-1)            (B, T, T), each row sums to 1
  out    = attn @ V                            (B, T, d_v)

We cache every intermediate needed for the backward pass (stage 4), since
recomputing them there would be wasteful and error-prone.
"""

import os

import numpy as np

from embeddings import embed, init_embeddings
from tokenizer import DATA_DIR, CharTokenizer


def init_attn_params(d_model, d_k, d_v, seed=0):
    rng = np.random.default_rng(seed)
    scale = 0.02
    return {
        "Wq": rng.normal(0, scale, size=(d_model, d_k)),
        "Wk": rng.normal(0, scale, size=(d_model, d_k)),
        "Wv": rng.normal(0, scale, size=(d_model, d_v)),
    }


def causal_mask(T):
    """ (T, T) bool array, True where attention is ALLOWED (j <= i). """
    return np.tril(np.ones((T, T), dtype=bool))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability, doesn't change result
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention_forward(x, params):
    """x: (B, T, d_model) -> out: (B, T, d_v), cache: dict of intermediates."""
    B, T, d_model = x.shape
    Wq, Wk, Wv = params["Wq"], params["Wk"], params["Wv"]
    d_k = Wq.shape[1]

    Q = x @ Wq  # (B, T, d_k)
    K = x @ Wk  # (B, T, d_k)
    V = x @ Wv  # (B, T, d_v)

    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)  # (B, T, T)

    mask = causal_mask(T)  # (T, T)
    masked_scores = np.where(mask[None, :, :], scores, -np.inf)

    attn = softmax(masked_scores, axis=-1)  # (B, T, T)
    out = attn @ V  # (B, T, d_v)

    cache = {
        "x": x, "Q": Q, "K": K, "V": V,
        "scores": scores, "mask": mask, "attn": attn,
        "d_k": d_k, "params": params,
    }
    return out, cache


def attention_backward(dout, cache):
    """Stage 4: backward pass through attention_forward.

    dout: (B, T, d_v) -- gradient of the loss w.r.t. `out`.
    Returns dx, dWq, dWk, dWv -- gradients w.r.t. every input/param that
    produced `out`, via the multivariate chain rule applied in reverse
    order to each op in the forward pass.
    """
    x, Q, K, V, attn = cache["x"], cache["Q"], cache["K"], cache["V"], cache["attn"]
    d_k = cache["d_k"]
    Wq, Wk, Wv = cache["params"]["Wq"], cache["params"]["Wk"], cache["params"]["Wv"]

    # out[b] = attn[b] @ V[b]   (T,T)@(T,d_v)
    dV = attn.transpose(0, 2, 1) @ dout          # (B, T, d_v)
    dattn = dout @ V.transpose(0, 2, 1)          # (B, T, T)

    # attn = softmax(masked_scores, axis=-1). Standard softmax jacobian,
    # applied row-wise: dscore_j = attn_j * (dattn_j - sum_k attn_k*dattn_k).
    # Masked positions have attn_j == 0 exactly, so they come out to 0 here
    # too -- no extra masking of the gradient needed.
    s = np.sum(attn * dattn, axis=-1, keepdims=True)   # (B, T, 1)
    dscores = attn * (dattn - s)                       # (B, T, T)

    # scores = Q @ K^T / sqrt(d_k)
    dscores = dscores / np.sqrt(d_k)
    dQ = dscores @ K                              # (B, T, d_k)
    dK = dscores.transpose(0, 2, 1) @ Q           # (B, T, d_k)

    # Q = x @ Wq, K = x @ Wk, V = x @ Wv  (weights shared across batch)
    dWq = np.einsum("btd,btk->dk", x, dQ)
    dWk = np.einsum("btd,btk->dk", x, dK)
    dWv = np.einsum("btd,btv->dv", x, dV)

    dx = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T        # (B, T, d_model)

    return dx, dWq, dWk, dWv


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "corpus.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)

    d_model, block_size, d_k, d_v = 8, 16, 8, 8
    tok_emb, pos_emb = init_embeddings(tok.vocab_size, block_size, d_model)
    attn_params = init_attn_params(d_model, d_k, d_v)

    batch_strs = ["the quick br", "og barks at "]
    ids = np.stack([tok.encode(s) for s in batch_strs])
    x = embed(ids, tok_emb, pos_emb)

    out, cache = attention_forward(x, attn_params)
    print(f"x shape:   {x.shape}")
    print(f"out shape: {out.shape}")

    attn = cache["attn"]
    print(f"\nattention weights for seq 0 (rows=query pos, cols=key pos):")
    with np.printoptions(precision=2, suppress=True):
        print(attn[0])

    # sanity check 1: each row sums to 1 (it's a probability distribution)
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0), row_sums
    print("\nrow-sums == 1 check OK")

    # sanity check 2: causal mask respected -- zero weight on future positions
    T = ids.shape[1]
    future_mask = ~causal_mask(T)
    assert np.allclose(attn[:, future_mask], 0.0)
    print("causal mask (no attention to future) check OK")

    # sanity check 3: position 0 can only attend to itself -> weight is exactly 1
    assert np.allclose(attn[:, 0, 0], 1.0)
    print("first-position self-attention check OK")

    # ---- Stage 4: backward pass + numerical gradient check ----
    from gradcheck import check_gradient, numerical_gradient

    print("\n--- gradient check: attention backward ---")

    rng = np.random.default_rng(42)
    dout_seed = rng.normal(size=out.shape)  # fixed "upstream gradient"

    # loss = sum(out * dout_seed) is a generic scalar functional of the
    # params/inputs; its gradient w.r.t. `out` is exactly dout_seed, which
    # lets us gradcheck attention_backward in isolation (no need for a real
    # downstream loss function yet -- that comes in stage 6).
    def loss_fn(x_, params_):
        out_, _ = attention_forward(x_, params_)
        return np.sum(out_ * dout_seed)

    dx, dWq, dWk, dWv = attention_backward(dout_seed, cache)

    x_work = x.copy()
    num_dx = numerical_gradient(lambda: loss_fn(x_work, attn_params), x_work)

    Wq_work = attn_params["Wq"].copy()
    params_work = dict(attn_params, Wq=Wq_work)
    num_dWq = numerical_gradient(lambda: loss_fn(x, params_work), Wq_work)

    Wk_work = attn_params["Wk"].copy()
    params_work2 = dict(attn_params, Wk=Wk_work)
    num_dWk = numerical_gradient(lambda: loss_fn(x, params_work2), Wk_work)

    Wv_work = attn_params["Wv"].copy()
    params_work3 = dict(attn_params, Wv=Wv_work)
    num_dWv = numerical_gradient(lambda: loss_fn(x, params_work3), Wv_work)

    ok = True
    ok &= check_gradient("dx", dx, num_dx)
    ok &= check_gradient("dWq", dWq, num_dWq)
    ok &= check_gradient("dWk", dWk, num_dWk)
    ok &= check_gradient("dWv", dWv, num_dWv)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- attention backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED -- do not proceed to stage 5.")
        raise SystemExit(1)
