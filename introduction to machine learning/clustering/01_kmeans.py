"""
K-Means Clustering.

Concept: picks K cluster centers ("centroids"), assigns every point to
its nearest centroid, then moves each centroid to the average position
of its assigned points, repeating until the assignments stop changing.

Note: the `approved` label is NOT used here -- clustering is unsupervised,
grouping rows purely by feature similarity. With only 2 rows, K=2 is the
only sensible choice, and it converges instantly: each point simply
becomes its own centroid.
"""

import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

model = KMeans(n_clusters=2, n_init=10, random_state=0)
model.fit(X)

print("=== K-Means internal state ===")
print("Learned centroids (one per cluster):")
print(pd.DataFrame(model.cluster_centers_, columns=X.columns))
print(f"\nIterations until convergence: {model.n_iter_}")
print(f"Inertia (sum of squared distances to nearest centroid): {model.inertia_:.4f}")
print("\nNOTE: with 2 points and 2 clusters, each point becomes its own "
      "centroid exactly, so inertia is 0 -- there's no averaging of "
      "multiple points into one centroid to observe here.")

print("\n=== Final cluster assignments ===")
labels = model.labels_
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {labels[i]}")
