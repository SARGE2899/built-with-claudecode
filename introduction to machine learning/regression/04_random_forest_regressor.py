"""
Random Forest Regression.

Concept: averages the predictions of many decision trees, each trained
on a random bootstrap resample of the data, to smooth out the
overfitting any single tree would show.

With only 2 rows, every bootstrap resample can only ever contain copies
of those same 2 points, so every tree in the forest learns the same
trivial split -- the "averaging many diverse trees" benefit of random
forests genuinely cannot show up with data this small, which we print
explicitly below rather than pretend the ensemble is doing something
more sophisticated.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

model = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
model.fit(X, y)

print("=== Random forest regressor internal state ===")
print(f"Number of trees: {len(model.estimators_)}")
print("Feature importances (averaged across trees):")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\nPer-tree prediction for row 0 (all trees see the same 2 possible "
      "points, so little diversity is possible with only 2 rows):")
tree_preds = [tree.predict(X.iloc[[0]])[0] for tree in model.estimators_]
print(f"  {tree_preds}")

print("\n=== Final predictions (averaged across all trees) ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
