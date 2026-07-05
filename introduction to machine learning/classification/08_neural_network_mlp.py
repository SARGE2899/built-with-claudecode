"""
Multi-Layer Perceptron (MLP) classification, via sklearn.

Concept: a small feedforward neural network -- an input layer, one or
more "hidden" layers of artificial neurons (each a weighted sum passed
through a nonlinearity), and an output layer -- trained end-to-end with
backpropagation to minimize prediction error. This is the sklearn
convenience version; neural_networks/01_single_perceptron.py builds the
single-neuron case by hand with raw numpy so the underlying math is
visible.

With only 2 training points, this tiny network will simply memorize
them; it needs many iterations because gradient-based training on 2
points has very little signal per step.
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

# Standardize features by hand (mean 0, std 1) -- MLPs are sensitive to
# feature scale since age (~tens) and credit_score (~hundreds) would
# otherwise dominate the initial gradients purely due to magnitude.
X_scaled = (X - X.mean()) / X.std()

model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    model.fit(X_scaled, y)

print("=== MLP internal state ===")
print(f"Network shape: {X_scaled.shape[1]} inputs -> "
      f"{model.hidden_layer_sizes[0]} hidden -> 1 output")

for layer_i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"\nLayer {layer_i} weight matrix, shape {weights.shape}:")
    print(weights)
    print(f"Layer {layer_i} biases: {biases}")

print("\n=== Final predictions ===")
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)
for i in range(len(X_scaled)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=1)={probs[i][1]:.4f}")
