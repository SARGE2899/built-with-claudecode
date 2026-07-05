"""Stage 5a: LayerNorm, forward + backward, gradchecked.

LayerNorm normalizes each token's feature vector (last axis) to zero mean,
unit variance, then applies a learned per-feature scale/shift (gamma/beta).
This keeps activations well-scaled as they flow through many stacked
blocks, which is what makes deep transformers trainable at all.

  mu    = mean(x, axis=-1)                    per-token mean
  var   = mean((x - mu)^2, axis=-1)           per-token (biased) variance
  xhat  = (x - mu) / sqrt(var + eps)          normalized
  y     = gamma * xhat + beta                 learned affine

Backward is the fiddly part: mu and var both depend on every element of
x, so d(xhat_i)/d(x_j) is dense, not diagonal. The closed form (derived
by writing out the chain rule over mu and var and simplifying) is:

  dxhat = dy * gamma
  dx    = (1/N) * inv_std * ( N*dxhat - sum(dxhat) - xhat * sum(dxhat*xhat) )

where N = d_model and the two sums are over the last axis, kept for
broadcasting. We verify this against numerical differentiation below
rather than trusting the algebra blindly.
"""

import numpy as np


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x: (..., N). gamma, beta: (N,). Normalizes over the last axis."""
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mu) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv_std
    y = gamma * xhat + beta

    cache = {"xhat": xhat, "inv_std": inv_std, "gamma": gamma}
    return y, cache


def layernorm_backward(dy, cache):
    xhat, inv_std, gamma = cache["xhat"], cache["inv_std"], cache["gamma"]
    N = xhat.shape[-1]

    dgamma = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    dbeta = np.sum(dy, axis=tuple(range(dy.ndim - 1)))

    dxhat = dy * gamma
    sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)
    sum_dxhat_xhat = np.sum(dxhat * xhat, axis=-1, keepdims=True)
    dx = (inv_std / N) * (N * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)

    return dx, dgamma, dbeta


if __name__ == "__main__":
    import sys

    from gradcheck import check_gradient, numerical_gradient

    rng = np.random.default_rng(0)
    B, T, N = 2, 5, 8
    x = rng.normal(size=(B, T, N))
    gamma = rng.normal(1.0, 0.1, size=(N,))
    beta = rng.normal(0.0, 0.1, size=(N,))

    y, cache = layernorm_forward(x, gamma, beta)
    print(f"x shape: {x.shape} -> y shape: {y.shape}")

    # sanity check: per-token mean ~0, var ~1 before the affine transform
    xhat = cache["xhat"]
    print(f"xhat per-token mean (should be ~0): {xhat.mean(axis=-1)[0]}")
    print(f"xhat per-token var  (should be ~1): {xhat.var(axis=-1)[0]}")

    print("\n--- gradient check: layernorm backward ---")
    dout_seed = rng.normal(size=y.shape)

    def loss_fn(x_, gamma_, beta_):
        y_, _ = layernorm_forward(x_, gamma_, beta_)
        return np.sum(y_ * dout_seed)

    dx, dgamma, dbeta = layernorm_backward(dout_seed, cache)

    x_work = x.copy()
    num_dx = numerical_gradient(lambda: loss_fn(x_work, gamma, beta), x_work)

    gamma_work = gamma.copy()
    num_dgamma = numerical_gradient(lambda: loss_fn(x, gamma_work, beta), gamma_work)

    beta_work = beta.copy()
    num_dbeta = numerical_gradient(lambda: loss_fn(x, gamma, beta_work), beta_work)

    ok = True
    ok &= check_gradient("dx", dx, num_dx)
    ok &= check_gradient("dgamma", dgamma, num_dgamma)
    ok &= check_gradient("dbeta", dbeta, num_dbeta)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- layernorm backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED.")
        sys.exit(1)
