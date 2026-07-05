"""
K-Nearest Neighbors (KNN) classification.

Concept: to classify a point, look at its K closest neighbors (by distance
in feature space) and vote on the majority class among them. There is no
training/learning step at all -- KNN just memorizes the data and does all
its "work" at prediction time.

With only 2 rows in this dataset, K can be at most 2 (there are only 2
neighbors to look at), so we use K=1: each point's nearest neighbor is
simply the *other* point.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X, y)

print("=== KNN internal state ===")
print("Stored training points (KNN's 'model' IS the raw data):")
print(X.assign(approved=y))

distances, indices = model.kneighbors(X)
print("\nNearest-neighbor lookup for each row:")
for i in range(len(X)):
    neighbor_idx = indices[i][0]
    print(f"  row {i} (age={X.iloc[i]['age']}) -> nearest neighbor is "
          f"row {neighbor_idx} (age={X.iloc[neighbor_idx]['age']}), "
          f"distance={distances[i][0]:.2f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
