"""
Multilayer Perceptron (MLP), via sklearn.

Concept: stacks multiple layers of the single-perceptron idea from
01_single_perceptron.py -- an input layer, one or more hidden layers of
neurons with nonlinear activations, and an output layer -- so the
network can represent nonlinear decision boundaries a single neuron
cannot. Trained end-to-end via backpropagation.

This duplicates classification/08_neural_network_mlp.py by design: this
folder is meant to show the perceptron -> MLP progression side by side,
while the classification folder catalogs it alongside unrelated
algorithms (KNN, SVM, etc.) for a different kind of comparison.
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

X_scaled = (X - X.mean()) / X.std()

model = MLPClassifier(hidden_layer_sizes=(4, 3), max_iter=5000, random_state=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    model.fit(X_scaled, y)

print("=== MLP internal state ===")
print(f"Network shape: {X_scaled.shape[1]} inputs -> "
      f"{model.hidden_layer_sizes[0]} hidden -> {model.hidden_layer_sizes[1]} hidden -> 1 output")

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
