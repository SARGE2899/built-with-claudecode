"""
XGBoost Regression.

Concept: same gradient-boosting idea as the classifier version -- builds
trees one at a time, each fit to the residual errors of the ensemble so
far -- but with squared-error loss instead of logistic loss.

As with the classifier (see classification/07_xgboost.py), XGBoost's
default min_child_weight=1 is compared against a per-leaf Hessian sum.
For squared-error loss the Hessian per sample is always exactly 1 (a
constant, unlike logistic loss), so with 1 sample per leaf the sum is
exactly 1 -- right at the default threshold. This is borderline enough
that we set min_child_weight=0 explicitly rather than rely on a
knife-edge default to permit splitting.
"""

import pandas as pd
from xgboost import XGBRegressor

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

model = XGBRegressor(n_estimators=10, max_depth=2, min_child_weight=0, random_state=0)
model.fit(X, y)

print("=== XGBoost regressor internal state ===")
print("Feature importances (gain-based):")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\nFirst boosted tree (text dump):")
booster = model.get_booster()
print(booster.get_dump()[0])

print("=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
