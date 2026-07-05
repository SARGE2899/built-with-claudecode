"""
Support Vector Machine (SVM) classification.

Concept: finds the hyperplane that separates the classes with the
maximum possible margin (distance to the nearest points of each class).
Only the points closest to that boundary -- the "support vectors" --
actually determine where the boundary goes; every other point could be
deleted without changing the model.

With only 2 points (one per class), both points ARE the support
vectors by necessity -- the maximum-margin hyperplane is just the
perpendicular bisector of the line segment between them.
"""

import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = SVC(kernel="linear")
model.fit(X, y)

print("=== SVM internal state ===")
print("Support vectors (the training points that define the boundary):")
print(pd.DataFrame(model.support_vectors_, columns=X.columns))
print(f"\nIndices of support vectors in training data: {model.support_.tolist()}")
print("(both rows are support vectors: with 2 points of 2 classes, "
      "both necessarily lie on the margin)")

print("\nLearned hyperplane weights (coef_):")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name}: {coef:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
