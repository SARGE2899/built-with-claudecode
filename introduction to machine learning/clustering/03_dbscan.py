"""
DBSCAN (Density-Based Spatial Clustering).

Concept: groups together points that are densely packed (many neighbors
within a radius `eps`), and marks points in low-density regions as
"noise" that belongs to no cluster. Unlike K-Means, it doesn't need to
be told how many clusters to expect, and it can find oddly-shaped
clusters.

With only 2 points, DBSCAN's default min_samples=5 can NEVER be
satisfied -- no point can have 5 neighbors when only 2 points exist in
total. This isn't a bug to work around with more data; it's exactly
what "density-based" means: density requires a minimum count, and 2
points cannot ever meet a threshold of 5.
"""

import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

print("=== Attempt 1: DBSCAN with default min_samples=5 ===")
default_model = DBSCAN(eps=50, min_samples=5)
default_labels = default_model.fit_predict(X)
print(f"Labels: {default_labels}  (-1 means 'noise', i.e. not in any cluster)")
print("Both points are labeled noise: DBSCAN needs a minimum neighborhood "
      "density (min_samples) that 2 total points cannot ever satisfy, "
      "no matter how the eps radius is tuned. This is the algorithm "
      "working exactly as designed, not an error.")

print("\n=== Attempt 2: min_samples lowered to 1, purely to illustrate the "
      "mechanism (not a meaningful density-based result with 2 points) ===")
demo_model = DBSCAN(eps=50, min_samples=1)
demo_labels = demo_model.fit_predict(X)
print(f"Labels: {demo_labels}")
print("With min_samples=1, every point trivially satisfies its own "
      "'neighborhood', so points within eps of each other merge into one "
      "cluster. This demonstrates the mechanics but defeats the actual "
      "purpose of DBSCAN (distinguishing dense regions from sparse ones), "
      "since a dataset this size has no sparse/dense contrast to find.")

print("\n=== Final cluster assignments (from the min_samples=5 default) ===")
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {default_labels[i]}")
