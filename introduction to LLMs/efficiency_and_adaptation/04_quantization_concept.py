"""
Quantization (concept).

Concept: model weights are normally stored as float32 (4 bytes per
number). Quantization represents the SAME weight matrix using far fewer
bits per number -- e.g. int8 (1 byte) or int4 (half a byte) -- by
linearly mapping the matrix's real-valued range onto a small fixed set
of integers, storing those integers plus one scale (and optionally a
zero-point) per matrix, and reconstructing an APPROXIMATION of the
original float32 values on demand: `dequantized = (quantized_int - zero_point) * scale`.
Fewer bits means less memory and (with the right hardware support)
faster computation, at the cost of some numerical precision -- this
script measures that tradeoff directly rather than just asserting it.

Uses the same d_model=8 shape as the frozen weight matrix in
`01_lora.py`, so the two scripts' matrices are directly comparable.
"""

import numpy as np

D_MODEL = 8
rng = np.random.default_rng(0)

# Stand-in for a real pretrained weight matrix, same shape/style as
# `01_lora.py`'s frozen W.
W_float32 = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL)).astype(np.float32)

print("=== Original float32 weight matrix ===")
print(np.round(W_float32, 4))
print(f"dtype: {W_float32.dtype}, shape: {W_float32.shape}")


def quantize(x, n_bits):
    """Linear (affine) quantization to n_bits unsigned integers."""
    qmin, qmax = 0, 2 ** n_bits - 1
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale
    x_q = np.clip(np.round(x / scale + zero_point), qmin, qmax)
    return x_q.astype(np.int32), scale, zero_point


def dequantize(x_q, scale, zero_point):
    return (x_q.astype(np.float32) - zero_point) * scale


print("\n=== Memory footprint comparison ===")
n_params = W_float32.size
bytes_float32 = n_params * 4        # 32 bits = 4 bytes per parameter
bytes_int8 = n_params * 1           # 8 bits = 1 byte per parameter
bytes_int4 = n_params * 0.5         # 4 bits = half a byte per parameter (packed 2-per-byte)
print(f"Parameters: {n_params}")
print(f"float32: {bytes_float32:.0f} bytes")
print(f"int8:    {bytes_int8:.0f} bytes  ({100 * bytes_int8 / bytes_float32:.0f}% of float32)")
print(f"int4:    {bytes_int4:.0f} bytes  ({100 * bytes_int4 / bytes_float32:.0f}% of float32)")

for n_bits, label in [(8, "int8"), (4, "int4")]:
    print(f"\n=== Quantizing to {label} ({n_bits} bits, "
          f"{2 ** n_bits} representable levels) ===")
    W_q, scale, zero_point = quantize(W_float32, n_bits)
    print(f"Scale: {scale:.6f}, zero_point: {zero_point:.4f}")
    print(f"Quantized integer matrix:\n{W_q}")

    W_dequantized = dequantize(W_q, scale, zero_point)
    print(f"\nDequantized (reconstructed float) matrix:\n{np.round(W_dequantized, 4)}")

    abs_error = np.abs(W_float32 - W_dequantized)
    print(f"\n=== Final output ({label}): numerical error introduced ===")
    print(f"Mean absolute error: {abs_error.mean():.6f}")
    print(f"Max absolute error:  {abs_error.max():.6f}")
    print(f"Original value range: [{W_float32.min():.4f}, {W_float32.max():.4f}]")

print("\n=== Summary: the precision/memory tradeoff ===")
print(f"{'format':>8s}  {'bytes':>8s}  {'levels':>8s}")
for n_bits, label, size in [(32, 'float32', bytes_float32), (8, 'int8', bytes_int8), (4, 'int4', bytes_int4)]:
    levels = 2 ** n_bits if n_bits < 32 else "continuous"
    print(f"{label:>8s}  {size:>8.0f}  {str(levels):>8s}")
print("\nNOTE: int4 uses a quarter of int8's memory, but has only 16 "
      "representable levels across the SAME value range int8 splits into "
      "256 levels -- each representable value is responsible for a much "
      "wider range of 'real' values, which is exactly why int4's "
      "reconstruction error above is larger than int8's. This is the "
      "actual tradeoff quantization makes: the more aggressively you "
      "shrink memory, the coarser your weights become -- real "
      "quantization schemes (e.g. per-channel scales, or mixed precision "
      "for sensitive layers) exist specifically to claw back some of "
      "this lost precision without giving up all the memory savings.")
