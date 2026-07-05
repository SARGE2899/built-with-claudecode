"""
Spectral Clustering.

Concept: builds a similarity graph between points (an affinity matrix --
here a Gaussian/RBF kernel of pairwise distances), then clusters using
the eigenvectors of that graph's Laplacian. This lets it find clusters
by graph connectivity rather than raw distance, so it can separate
oddly-shaped clusters that distance-based methods like K-Means can't.

With only 2 points that differ this much (age 25 vs 55, income 80 vs
20, etc.), the default RBF kernel width considers them too far apart to
be "similar" at all -- their off-diagonal affinity rounds to 0.0, which
means the similarity graph has literally no edge between the only two
nodes it has. sklearn warns that the graph isn't fully connected rather
than erroring out; the eigendecomposition still runs and still recovers
the (here, trivially obvious) 2-way split, but there was no graph
structure for it to meaningfully exploit.
"""

import warnings

import pandas as pd
from sklearn.cluster import SpectralClustering

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = SpectralClustering(n_clusters=2, affinity="rbf", random_state=0)
    labels = model.fit_predict(X)
    if caught:
        print("=== Warnings sklearn raised (not errors -- it still ran) ===")
        for w in caught:
            print(f"  {w.category.__name__}: {w.message}")
        print()

print("=== Spectral Clustering internal state ===")
print("Affinity matrix (pairwise RBF similarity):")
print(pd.DataFrame(model.affinity_matrix_, index=["row0", "row1"], columns=["row0", "row1"]))
print("\nNOTE: the off-diagonal affinity is ~0.0 -- the default RBF kernel "
      "width sees these two rows as maximally dissimilar, so the "
      "similarity graph has no real edge between them. The graph-Laplacian "
      "eigendecomposition spectral clustering relies on has nothing richer "
      "to work with here; it recovers the 2-way split, but not because it "
      "found interesting graph structure -- there wasn't any to find.")

print("\n=== Final cluster assignments ===")
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {labels[i]}")
