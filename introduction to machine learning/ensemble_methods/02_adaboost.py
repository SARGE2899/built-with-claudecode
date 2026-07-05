"""
AdaBoost (Adaptive Boosting).

Concept: trains base models one at a time in sequence. After each round,
misclassified points get their sample weight increased, so the NEXT
model is forced to focus more on the examples the previous ones got
wrong. Final prediction is a weighted vote, where more-accurate models
in the sequence get more say.

With only 2 perfectly-separable points, the very first weak learner
already classifies both correctly -- there are no mistakes left to
"adapt" to, so every subsequent round just reinforces the same trivial
split rather than demonstrating boosting's real strength (incrementally
fixing hard cases).
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5, random_state=0,
)
model.fit(X, y)

print("=== AdaBoost internal state ===")
print(f"Per-estimator weight (how much say each gets in the final vote): "
      f"{model.estimator_weights_}")
print(f"Per-estimator training error: {model.estimator_errors_}")
print("\nNOTE: errors are 0 (or defaulted) from round 1 onward, since the "
      "single split available perfectly separates 2 points -- there is no "
      "residual mistake for later rounds to adaptively focus on.")

print("\n=== Final predictions (weighted vote across all rounds) ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
