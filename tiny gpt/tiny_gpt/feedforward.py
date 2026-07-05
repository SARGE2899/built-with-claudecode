"""Stage 5b: position-wise feedforward network, forward + backward, gradchecked.

Applied independently to every position (no mixing across time -- that's
attention's job). Standard transformer FFN: expand to a wider hidden dim,
apply a nonlinearity, project back down.

  z1 = x @ W1 + b1        (..., d_ff)   d_ff is usually 4*d_model
  h  = gelu(z1)
  y  = h @ W2 + b2        (..., d_model)

We use the tanh approximation of GELU (as in GPT-2), since exact GELU
needs erf and we're keeping this to plain NumPy:

  gelu(z) = 0.5*z*(1 + tanh(c*(z + 0.044715*z^3))),   c = sqrt(2/pi)

Its derivative (needed for backward) is:

  u = c*(z + 0.044715*z^3),  du/dz = c*(1 + 3*0.044715*z^2)
  gelu'(z) = 0.5*(1+tanh(u)) + 0.5*z*(1-tanh(u)^2)*du/dz
"""

import numpy as np

GELU_C = np.sqrt(2.0 / np.pi)


def gelu(z):
    u = GELU_C * (z + 0.044715 * z ** 3)
    return 0.5 * z * (1.0 + np.tanh(u))


def gelu_grad(z):
    u = GELU_C * (z + 0.044715 * z ** 3)
    tanh_u = np.tanh(u)
    du_dz = GELU_C * (1.0 + 3 * 0.044715 * z ** 2)
    return 0.5 * (1.0 + tanh_u) + 0.5 * z * (1.0 - tanh_u ** 2) * du_dz


def init_ffn_params(d_model, d_ff, seed=0):
    rng = np.random.default_rng(seed)
    scale = 0.02
    return {
        "W1": rng.normal(0, scale, size=(d_model, d_ff)),
        "b1": np.zeros(d_ff),
        "W2": rng.normal(0, scale, size=(d_ff, d_model)),
        "b2": np.zeros(d_model),
    }


def ffn_forward(x, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = x @ W1 + b1
    h = gelu(z1)
    y = h @ W2 + b2
    cache = {"x": x, "z1": z1, "h": h, "params": params}
    return y, cache


def ffn_backward(dy, cache):
    x, z1, h, params = cache["x"], cache["z1"], cache["h"], cache["params"]
    W1, W2 = params["W1"], params["W2"]

    sum_axes = tuple(range(dy.ndim - 1))
    dW2 = np.einsum("bti,btj->ij", h, dy)
    db2 = np.sum(dy, axis=sum_axes)

    dh = dy @ W2.T
    dz1 = dh * gelu_grad(z1)

    dW1 = np.einsum("bti,btj->ij", x, dz1)
    db1 = np.sum(dz1, axis=sum_axes)
    dx = dz1 @ W1.T

    return dx, dW1, db1, dW2, db2


if __name__ == "__main__":
    import sys

    from gradcheck import check_gradient, numerical_gradient

    rng = np.random.default_rng(0)
    B, T, d_model, d_ff = 2, 5, 8, 32
    x = rng.normal(size=(B, T, d_model))
    params = init_ffn_params(d_model, d_ff)
    # nonzero biases so gradcheck actually exercises them
    params["b1"] = rng.normal(0, 0.02, size=d_ff)
    params["b2"] = rng.normal(0, 0.02, size=d_model)

    y, cache = ffn_forward(x, params)
    print(f"x shape: {x.shape} -> y shape: {y.shape}")

    print("\n--- gradient check: feedforward backward ---")
    dout_seed = rng.normal(size=y.shape)

    def loss_fn(x_, params_):
        y_, _ = ffn_forward(x_, params_)
        return np.sum(y_ * dout_seed)

    dx, dW1, db1, dW2, db2 = ffn_backward(dout_seed, cache)

    checks = []

    x_work = x.copy()
    num_dx = numerical_gradient(lambda: loss_fn(x_work, params), x_work)
    checks.append(("dx", dx, num_dx))

    for name, key in [("dW1", "W1"), ("db1", "b1"), ("dW2", "W2"), ("db2", "b2")]:
        work = params[key].copy()
        params_work = dict(params, **{key: work})
        num = numerical_gradient(lambda: loss_fn(x, params_work), work)
        analytical = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}[name]
        checks.append((name, analytical, num))

    ok = True
    for name, analytical, num in checks:
        ok &= check_gradient(name, analytical, num)

    if ok:
        print("\nALL GRADIENT CHECKS PASSED -- feedforward backward is correct.")
    else:
        print("\nGRADIENT CHECK FAILED.")
        sys.exit(1)
