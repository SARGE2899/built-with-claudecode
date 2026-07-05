"""
Gradient Boosting.

Concept: like AdaBoost, builds models sequentially, but instead of
reweighting misclassified samples, each new tree is trained to predict
the *gradient of the loss* (roughly: the direction and size of the
current error) left by the ensemble so far, and its output is added in
to gradually reduce that error. This is the same idea XGBoost implements
with extra engineering (regularization, speed) on top.

With 2 perfectly separable points, the first tree already drives the
loss very low, so later trees are correcting an already-tiny residual --
you can watch `train_score_` (the loss per boosting stage) collapse to
~0 almost immediately.
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)
model.fit(X, y)

print("=== Gradient boosting internal state ===")
print("Feature importances:")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")
print(f"\nTraining loss (deviance) at each boosting stage:")
for i, score in enumerate(model.train_score_):
    print(f"  stage {i}: {score:.6f}")
print("\nNOTE: loss collapses almost immediately since 2 perfectly "
      "separable points give the very first tree everything it needs.")

print("\n=== Final predictions ===")
preds = model.predict(X)
probs = model.predict_proba(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=1)={probs[i][1]:.4f}")
