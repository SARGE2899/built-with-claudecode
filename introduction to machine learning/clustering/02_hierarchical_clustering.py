"""
Hierarchical (Agglomerative) Clustering.

Concept: starts with every point as its own cluster, then repeatedly
merges the two closest clusters together, building a tree of merges
(a "dendrogram") from the bottom up. Cutting the tree at a chosen height
gives you a specific number of clusters.

With only 2 points there is exactly one possible merge (combining the
two into one cluster) and therefore no real dendrogram structure to
explore -- we print the single linkage distance and say so explicitly,
rather than pretend a rich hierarchy exists.
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

Z = linkage(X, method="average")
print("=== Hierarchical clustering internal state (linkage matrix) ===")
print("Format: [cluster_a_idx, cluster_b_idx, distance, num_points_merged]")
print(Z)
print(f"\nThe single merge combines row 0 and row 1 at distance {Z[0][2]:.4f}.")
print("NOTE: with only 2 points there is exactly ONE possible merge -- "
      "a real dendrogram (showing which sub-groups merge before others) "
      "needs at least 3-4 points to have any structure to show.")

model = AgglomerativeClustering(n_clusters=2, linkage="average")
labels = model.fit_predict(X)

print("\n=== Final cluster assignments (cut to 2 clusters) ===")
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {labels[i]}")
