"""
Mean Shift Clustering.

Concept: place a "window" (radius = bandwidth) around every point, and
iteratively shift each window toward the average position of the points
inside it, until it converges to a local density peak. Points whose
windows converge to the same peak become one cluster -- like K-Means,
but the number of clusters emerges from the data instead of being
chosen up front.

With only 2 points, the bandwidth alone decides the outcome: too large
and both points' windows converge to the same peak (1 cluster); smaller
than the distance between them and each point's window never moves (2
clusters, each a singleton). There's no in-between "density landscape"
to discover with just 2 points, unlike real Mean Shift use cases with
hundreds of points forming actual density peaks -- sklearn's own
bandwidth estimator, built for exactly that many-point case, is used
here anyway just to see what it does at this extreme.
"""

import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

bandwidth = estimate_bandwidth(X, quantile=0.5)

print("=== Mean Shift internal state ===")
print(f"Estimated bandwidth (quantile=0.5): {bandwidth:.4f}")
if bandwidth <= 0:
    print("Estimator returned a non-positive bandwidth (can happen with "
          "this few points) -- falling back to a small positive value so "
          "the algorithm has something to work with.")
    bandwidth = 1.0

model = MeanShift(bandwidth=bandwidth)
model.fit(X)

print("\nLearned cluster centers:")
print(pd.DataFrame(model.cluster_centers_, columns=X.columns))
print(f"Number of clusters found: {len(model.cluster_centers_)}")

print("\n=== Final cluster assignments ===")
labels = model.labels_
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {labels[i]}")
