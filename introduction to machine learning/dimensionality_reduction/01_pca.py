"""
Principal Component Analysis (PCA).

Concept: finds new axes (linear combinations of the original features)
ordered by how much variance in the data they capture, so you can keep
just the first few axes and discard the rest with minimal information
loss. Unsupervised -- it never looks at the `approved` label.

With 2 data points, centering the data (subtracting the mean) leaves
exactly 2 vectors that are negatives of each other -- mathematically,
that spans only 1 dimension, no matter how many original features (5)
there were. So while sklearn happily returns 2 components here, the
second one captures ~0% of the variance: we're asking for a "2D
summary" of something that is intrinsically 1-dimensional.
"""

import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

model = PCA(n_components=2)
transformed = model.fit_transform(X)

print("=== PCA internal state ===")
print("Principal component vectors (rows = components, in original feature space):")
print(pd.DataFrame(model.components_, columns=X.columns,
                    index=["PC1", "PC2"]))
print(f"\nExplained variance ratio: {model.explained_variance_ratio_}")
print("NOTE: PC2's explained variance is ~0 (numerically, not exactly 0 due "
      "to floating point). With only 2 points, the centered data spans "
      "exactly 1 dimension -- PC1 captures ALL the real structure, and "
      "PC2 is just an arbitrary direction orthogonal to it, carrying no "
      "information.")

print("\n=== Final transformed coordinates (5 features -> 2D) ===")
for i in range(len(X)):
    print(f"  row {i}: {transformed[i]}")
