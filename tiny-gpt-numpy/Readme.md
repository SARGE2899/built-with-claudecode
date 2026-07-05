# tiny_gpt

A minimal GPT-style transformer language model, built entirely from scratch in **pure NumPy** — no PyTorch, no autograd. Every forward pass and every backward pass (gradient) is hand-derived and verified numerically.

This project exists to actually understand how a transformer works internally, rather than treating `.backward()` as magic.

## What's inside

| File | Purpose |
|---|---|
| `tokenizer.py` | Character-level tokenizer — text ↔ integer ids |
| `embeddings.py` | Token embeddings + positional embeddings |
| `attention.py` | Single-head causal self-attention (forward + backward) |
| `mha.py` | Multi-head attention |
| `layernorm.py` | Layer normalization (forward + backward) |
| `feedforward.py` | Position-wise feedforward network |
| `block.py` | Full transformer block (attention + FFN + LayerNorm + residuals) |
| `model.py` | Stacks blocks into the complete model + cross-entropy loss |
| `train.py` | Training loop (forward, backward, Adam optimizer update) |
| `generate.py` | Autoregressive text generation with temperature sampling |
| `gradcheck.py` | Numerical gradient checking utilities |
| `run_all_tests.py` | Runs every stage's gradient check end-to-end |

## Why it matters

Every backward pass is verified against numerical (finite-difference) gradients before being trusted — relative errors in the `1e-06` to `1e-11` range across every layer, including the full stacked model with cross-entropy loss. This isn't "it runs without crashing" — it's mathematically confirmed correct backpropagation, written by hand.

## Running it

```bash
# Verify every layer's gradients are correct
python run_all_tests.py

# Train on the sample corpus
python train.py

# Train + generate text
python generate.py
```

## Current status

Trained (and currently overfits, as expected) on a small sample corpus. At low sampling temperature it reproduces the training text closely; at higher temperature it explores more and starts to break down — expected behavior for a tiny model trained on a tiny corpus.

Next step: train on a larger corpus to see actual generalization rather than memorization.

## Stack

Pure Python + NumPy. No ML frameworks. Built stage-by-stage with gradient verification gating progress at every step.
