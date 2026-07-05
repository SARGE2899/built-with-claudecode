"""
t-SNE (t-Distributed Stochastic Neighbor Embedding).

Concept: models how likely each point is to be a "neighbor" of every
other point (based on distance) in the original high-dimensional space,
then finds a low-dimensional layout where those neighbor probabilities
are preserved as closely as possible. It's designed to reveal cluster
structure among MANY points by comparing local neighborhoods.

The `perplexity` parameter roughly says "how many neighbors should each
point weigh" -- sklearn requires perplexity < n_samples, so with only 2
points, perplexity must be < 2 (we use 1). But with just 2 points there
is no neighborhood structure to model in the first place: t-SNE reduces
to a coin-flip of which point goes left/right, with the actual distance
between them in the embedding being fairly arbitrary. This is not a
representative use of the algorithm, just a mechanical demonstration.
"""

import pandas as pd
from sklearn.manifold import TSNE

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

model = TSNE(n_components=2, perplexity=1, random_state=0)
transformed = model.fit_transform(X)

print("=== t-SNE internal state ===")
print(f"KL divergence of final embedding: {model.kl_divergence_:.6f}")
print("NOTE: t-SNE has no reusable 'model' like PCA's component vectors -- "
      "it directly optimizes the low-dimensional coordinates for THIS "
      "dataset only, and cannot transform new/unseen points. There is no "
      "coefficient matrix to print, only the resulting layout.")
print("\nWe set perplexity=1 because sklearn requires perplexity < "
      "n_samples (2 here); with only 2 points there are no meaningful "
      "'neighborhoods' to preserve, so the specific embedding coordinates "
      "below carry little of the meaning they would with real data.")

print("\n=== Final transformed coordinates (5 features -> 2D) ===")
for i in range(len(X)):
    print(f"  row {i}: {transformed[i]}")
