"""
LoRA (Low-Rank Adaptation).

Concept: fully fine-tuning an LLM means updating every parameter in
every weight matrix -- expensive in both compute and storage (you'd need
a full separate copy of every adapted matrix per task/user). LoRA
freezes the original pretrained weight matrix W entirely (it never
changes) and instead learns a much smaller UPDATE to it, expressed as
the product of two skinny matrices: A (d_model x r) and B (r x d_model),
where the rank r is chosen far smaller than d_model. The adapted weight
used at inference is:

    W_adapted = W_frozen + (alpha / r) * (A @ B)

This works because meaningful fine-tuning updates tend to have low
"intrinsic rank" -- they don't need to move the weight matrix in every
possible direction, just a few useful ones -- so a rank-r approximation
captures most of the useful adaptation while training/storing
dramatically fewer parameters than the full matrix.

Uses d_model=8 (the same dimension as every attention/architecture
script in this project) as the frozen matrix's shape, so it's directly
comparable to e.g. `attention/01_scaled_dot_product_attention.py`'s
W_q.
"""

import numpy as np

D_MODEL = 8
RANK = 2       # r -- deliberately much smaller than d_model=8
ALPHA = 4      # LoRA scaling factor; effective scale applied is ALPHA/RANK

rng = np.random.default_rng(0)

# Stand-in for a PRETRAINED weight matrix (e.g. an attention W_q) -- this
# is exactly what LoRA freezes and never updates.
W_frozen = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_frozen_original_copy = W_frozen.copy()  # saved so we can later PROVE it never changed

print("=== Frozen pretrained weight ===")
print(f"W_frozen shape: {W_frozen.shape}")
print(f"Total parameters in W_frozen: {W_frozen.size} (all frozen -- zero of these are trained)")

# LoRA's own initialization convention matters: A is small random noise,
# but B starts at EXACTLY ZERO -- so at the very start of fine-tuning,
# A @ B = 0 and the adapted weight is numerically IDENTICAL to the
# original frozen weight. Training then moves B (and A) away from this
# starting point.
A = rng.normal(scale=0.1, size=(D_MODEL, RANK))
B = np.zeros((RANK, D_MODEL))

print(f"\n=== LoRA matrices at initialization ===")
print(f"A shape: {A.shape} (d_model x r)")
print(f"B shape: {B.shape} (r x d_model) -- initialized to all zeros")

delta_at_init = (ALPHA / RANK) * (A @ B)
W_adapted_at_init = W_frozen + delta_at_init

print(f"\nDelta (A @ B, scaled by alpha/r={ALPHA}/{RANK}={ALPHA/RANK}) at init:")
print(np.round(delta_at_init, 6))
print(f"W_adapted == W_frozen exactly at init? "
      f"{np.array_equal(W_adapted_at_init, W_frozen)}")
print("NOTE: this zero-initialization is deliberate -- it guarantees "
      "fine-tuning STARTS from behavior identical to the original "
      "pretrained model, and only diverges as A and B are actually "
      "trained, rather than starting from a random perturbation of a "
      "perfectly good pretrained weight.")

print("\n=== Simulating a training step (B moves away from zero) ===")
# A real training step would compute this via backpropagation through a
# loss; here we simply perturb B by hand to demonstrate the MECHANISM --
# what changes, and, just as importantly, what does NOT change.
B_after_step = B + rng.normal(scale=0.05, size=B.shape)
delta_after_step = (ALPHA / RANK) * (A @ B_after_step)
W_adapted_after_step = W_frozen + delta_after_step

print(f"B after one (simulated) training step:\n{np.round(B_after_step, 4)}")
print(f"\nNew delta (A @ B, scaled):\n{np.round(delta_after_step, 4)}")

print(f"\n=== Final output: adapted weight vs. frozen original ===")
print(f"W_frozen still bit-for-bit identical to its value before any "
      f"'training' happened? {np.array_equal(W_frozen, W_frozen_original_copy)} "
      f"-- nothing in the LoRA update process above ever touched it.")
print(f"W_adapted = W_frozen + delta:")
print(np.round(W_adapted_after_step, 4))
print(f"Max absolute difference from W_frozen: "
      f"{np.max(np.abs(W_adapted_after_step - W_frozen)):.4f}")

print("\n=== Parameter count: LoRA vs. full fine-tuning ===")
full_finetune_params = D_MODEL * D_MODEL
lora_params = A.size + B.size
print(f"Full fine-tuning would update: {full_finetune_params} parameters "
      f"(the entire W matrix)")
print(f"LoRA updates only: {lora_params} parameters "
      f"(A: {A.size} + B: {B.size}), "
      f"{100 * lora_params / full_finetune_params:.1f}% of full fine-tuning")
print(f"\nAt this tiny d_model=8 scale the saving looks modest, but the "
      f"parameter count for full fine-tuning grows as d_model^2 while "
      f"LoRA's grows as 2 * d_model * r -- at GPT-3 scale (d_model in the "
      f"thousands, r often 4-16), that quadratic-vs-linear gap is what "
      f"makes LoRA practical at all: fine-tuning becomes cheap enough to "
      f"store a separate small adapter per task instead of a full copy "
      f"of the entire model per task.")
