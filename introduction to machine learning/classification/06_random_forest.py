"""
Random Forest classification.

Concept: trains many decision trees, each on a random bootstrap sample of
the data (sampling with replacement) and each considering only a random
subset of features at every split, then has them vote on the final
prediction. Averaging many "noisy" trees reduces overfitting compared to
a single tree.

With only 2 rows, bootstrap sampling can easily produce a resample
containing the same row twice (e.g. both class 0) -- some trees in the
forest may only ever see one class. This is normal here and simply
means those trees always predict that one class.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
model.fit(X, y)

print("=== Random forest internal state ===")
print(f"Number of trees: {len(model.estimators_)}")
print("Feature importances (averaged across all trees):")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\nWhat each individual tree in the forest predicts for row 0:")
for i, tree in enumerate(model.estimators_):
    pred = tree.predict(X.iloc[[0]])[0]
    print(f"  tree {i}: predicts {pred}")

print("\n=== Final predictions (majority vote across all trees) ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
