"""
Decision Tree classification.

Concept: recursively splits the data on the single feature/threshold that
best separates the classes (using a purity measure like Gini impurity),
building a tree of if/else rules. Prediction is just walking the tree
from root to leaf following those rules.

With 2 perfectly separable points, literally any one feature threshold
separates them completely, so the tree ends up with a single split
(depth 1) and 100% purity.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

print("=== Decision tree internal state (learned split rules) ===")
print(export_text(model, feature_names=list(X.columns)))

print("Feature importances:")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
