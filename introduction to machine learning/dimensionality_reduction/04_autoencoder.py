"""
Autoencoder (tiny neural network for dimensionality reduction).

Concept: a neural network trained to reproduce its own input, but forced
through a narrow "bottleneck" layer in the middle (here, 2 units). To
reconstruct the input well despite that bottleneck, the network must
learn a compressed 2-number encoding that captures whatever's most
useful about each 5-feature row -- the bottleneck activations ARE the
2D representation, analogous to PCA's transformed coordinates but
learned via gradient descent instead of linear algebra.

Unsupervised, like PCA/t-SNE -- the `approved` label is not used. With
only 2 training rows, this network can trivially memorize a perfect
reconstruction (it has far more parameters than data points), so this
demonstrates the mechanism clearly but says nothing about how well it
would generalize to unseen rows.
"""

import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

# Standardize features by hand -- gradient descent on raw scales
# (age ~tens, credit_score ~hundreds) would be dominated by the
# largest-magnitude feature otherwise.
X_mean, X_std = X.mean(), X.std()
X_scaled = (X - X_mean) / X_std
X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)

n_features = X_tensor.shape[1]
bottleneck = 2


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(n_features, bottleneck)
        self.decoder = nn.Linear(bottleneck, n_features)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


model = Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(500):
    optimizer.zero_grad()
    reconstructed, _ = model(X_tensor)
    loss = loss_fn(reconstructed, X_tensor)
    loss.backward()
    optimizer.step()

print("=== Autoencoder internal state (learned weights) ===")
print("Encoder weight matrix (5 features -> 2D bottleneck):")
print(model.encoder.weight.detach().numpy())
print("Encoder bias:", model.encoder.bias.detach().numpy())
print("\nDecoder weight matrix (2D bottleneck -> 5 features):")
print(model.decoder.weight.detach().numpy())
print("Decoder bias:", model.decoder.bias.detach().numpy())

with torch.no_grad():
    reconstructed, latent = model(X_tensor)
    final_loss = loss_fn(reconstructed, X_tensor).item()

print(f"\nFinal reconstruction MSE (standardized scale): {final_loss:.6f}")

print("\n=== Final transformed coordinates (5 features -> 2D bottleneck) ===")
for i in range(len(X)):
    print(f"  row {i}: {latent[i].numpy()}")

print("\n=== Reconstruction quality (standardized scale) ===")
for i in range(len(X)):
    print(f"  row {i} original:      {X_tensor[i].numpy()}")
    print(f"  row {i} reconstructed: {reconstructed[i].numpy()}")
