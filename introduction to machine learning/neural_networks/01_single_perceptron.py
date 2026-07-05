"""
Single Perceptron (hand-rolled, no sklearn).

Concept: the simplest possible neural network -- one artificial neuron.
It computes a weighted sum of the inputs plus a bias (w . x + b), then
squashes that through a sigmoid function to output a probability
between 0 and 1. This is mathematically IDENTICAL to logistic
regression; the "neural network" framing is just the lens of viewing it
as one neuron, which is the building block the MLP in the next script
stacks many of together.

Every piece here (sigmoid, the forward pass, the gradient, the update
rule) is implemented by hand with plain numpy, so the raw mechanics
that libraries usually hide are fully visible.
"""

import numpy as np
import pandas as pd

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"]).values.astype(float)
y = df["approved"].values.astype(float)

# Standardize features by hand -- without this, credit_score (~hundreds)
# would dominate the weighted sum purely due to its scale, swamping the
# gradient contributions of smaller-scale features like years_employed.
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_scaled = (X - X_mean) / X_std

n_features = X_scaled.shape[1]
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, size=n_features)  # weights, one per feature
b = 0.0                                   # bias


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


lr = 0.5
n_epochs = 2000
for epoch in range(n_epochs):
    # forward pass
    z = X_scaled @ w + b          # weighted sum: w . x + b, for all rows at once
    y_hat = sigmoid(z)             # squashed to a probability

    # binary cross-entropy loss gradient w.r.t. z simplifies to (y_hat - y)
    # for sigmoid + cross-entropy specifically -- this is the one piece of
    # calculus we're relying on, everything else is direct arithmetic.
    error = y_hat - y

    # backward pass: gradient of the loss w.r.t. w and b
    n = len(y)
    dw = (X_scaled.T @ error) / n
    db = np.sum(error) / n

    # gradient descent update
    w -= lr * dw
    b -= lr * db

print("=== Single perceptron internal state (hand-computed) ===")
print("Learned weights (one per standardized feature):")
for name, weight in zip(df.drop(columns=["approved"]).columns, w):
    print(f"  {name}: {weight:.4f}")
print(f"Learned bias: {b:.4f}")

print("\n=== Final predictions ===")
final_z = X_scaled @ w + b
final_probs = sigmoid(final_z)
final_preds = (final_probs >= 0.5).astype(int)
for i in range(len(y)):
    print(f"  row {i}: true={int(y[i])}  predicted={final_preds[i]}  "
          f"P(approved)={final_probs[i]:.6f}")
