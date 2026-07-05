"""
Support Vector Regression (SVR).

Concept: like SVM classification, but instead of maximizing margin
between classes, it fits a function that tries to keep predictions
within a tolerance tube (epsilon) of the true values, only penalizing
points that fall outside that tube. Points outside (or on the edge of)
the tube become the "support vectors" that define the fitted function.

With only 2 points, both fall outside any nonzero-width tube around a
single flat prediction, so both end up as support vectors -- there is
no "easy majority" for SVR to ignore.
"""

import pandas as pd
from sklearn.svm import SVR

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

model = SVR(kernel="linear", epsilon=0.5)
model.fit(X, y)

print("=== SVR internal state ===")
print("Support vectors (rows that fall outside the epsilon tolerance tube):")
print(pd.DataFrame(model.support_vectors_, columns=X.columns))
print(f"Dual coefficients: {model.dual_coef_}")

print("\nLearned hyperplane weights (linear kernel coef_):")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name}: {coef:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
