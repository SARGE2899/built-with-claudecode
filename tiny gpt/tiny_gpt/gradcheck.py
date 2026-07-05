"""Generic numerical gradient checking, reused at every backward-pass stage.

The idea: for a scalar-valued function f(param), the definition of the
derivative gives us a way to *estimate* it without any calculus --
just evaluate f at param+eps and param-eps and take the slope:

    df/dparam[i] ~= (f(param + eps*e_i) - f(param - eps*e_i)) / (2*eps)

This central-difference estimate has O(eps^2) error (vs O(eps) for a
one-sided difference), which is why we use it as ground truth to check
our hand-derived analytical (backprop) gradients against.
"""

import numpy as np


def numerical_gradient(f, param, eps=1e-5):
    """f: callable taking the array `param` (with this exact value substituted
    in) and returning a scalar float. Returns an array the same shape as
    `param` containing the estimated gradient df/dparam.
    """
    grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        f_plus = f()

        param[idx] = orig - eps
        f_minus = f()

        param[idx] = orig  # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a, b, eps=1e-8):
    """Elementwise relative error, safe when both a and b are near zero."""
    return np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b))


def walk_params(params, grads, prefix=""):
    """Recursively walk parallel nested dict/list structures of params and
    grads (e.g. a model with a list of blocks, each holding nested MHA/FFN
    param dicts), yielding (path, param_array, grad_array) for every
    ndarray leaf. Non-array leaves (like an `n_heads` int) are skipped.
    """
    if isinstance(params, dict):
        for k in grads:  # iterate grads' keys: params may hold extra non-array
            yield from walk_params(params[k], grads[k], f"{prefix}.{k}" if prefix else k)
    elif isinstance(params, list):
        for i, (p, g) in enumerate(zip(params, grads)):
            yield from walk_params(p, g, f"{prefix}[{i}]")
    elif isinstance(params, np.ndarray):
        yield prefix, params, grads


def check_gradient(name, analytical, numerical, tol=1e-4, atol=1e-6, verbose=True):
    """Compare analytical vs numerical gradients. An element passes if its
    relative error is small OR its absolute error is small (mirrors
    np.allclose's atol+rtol combination). Pure relative error is a bad
    metric near zero: two gradients of ~1e-8 that agree to 3 significant
    figures can show a "huge" relative error purely from finite-difference
    and float64 rounding noise, even though nothing is wrong.
    """
    abs_diff = np.abs(analytical - numerical)
    rel_err = relative_error(analytical, numerical)
    fail_mask = (rel_err >= tol) & (abs_diff >= atol)

    passed = not np.any(fail_mask)
    max_err = np.max(rel_err[fail_mask]) if np.any(fail_mask) else np.max(rel_err)
    status = "PASS" if passed else "FAIL"
    if verbose:
        print(f"  [{status}] {name}: max relative error = {max_err:.2e} "
              f"(shape {analytical.shape})")
        if not passed:
            worst = np.unravel_index(np.argmax(np.where(fail_mask, rel_err, -1)), rel_err.shape)
            print(f"         worst at index {worst}: "
                  f"analytical={analytical[worst]:.6e}, "
                  f"numerical={numerical[worst]:.6e}, "
                  f"abs_diff={abs_diff[worst]:.2e}")
    return passed
