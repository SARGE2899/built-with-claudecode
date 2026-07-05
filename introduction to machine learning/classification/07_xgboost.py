"""
XGBoost classification.

Concept: builds an ensemble of decision trees one at a time, where each
new tree is trained specifically to correct the errors (residual
gradients) of the ensemble built so far ("gradient boosting"), rather
than all trees being trained independently like in a random forest.

With only 2 rows, each boosting round essentially just double-checks
that the single already-obvious split (whichever feature separates the
two classes) is being reinforced -- there's no room for the subtlety
gradient boosting is normally used for.

IMPORTANT gotcha found while building this: XGBoost's default
min_child_weight=1 is compared against the *sum of Hessians* in a
candidate leaf, not the sample count. For logistic loss the per-sample
Hessian is p*(1-p), which maxes out at 0.25 -- so a leaf with only 1
sample always has Hessian-sum <= 0.25 < 1, and XGBoost's default refuses
to split at all, silently producing a single-leaf "model" that predicts
0.5 for everything. We set min_child_weight=0 below purely to let it
split with this few samples; this is not a hack to fabricate a result,
it's disabling a regularization default that assumes far more data.
"""

import pandas as pd
from xgboost import XGBClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = XGBClassifier(n_estimators=10, max_depth=2, min_child_weight=0,
                       eval_metric="logloss", random_state=0)
model.fit(X, y)

print("=== XGBoost internal state ===")
print("Feature importances (gain-based):")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\nFirst boosted tree (text dump):")
booster = model.get_booster()
print(booster.get_dump()[0])

print("=== Final predictions ===")
preds = model.predict(X)
probs = model.predict_proba(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=1)={probs[i][1]:.4f}")
