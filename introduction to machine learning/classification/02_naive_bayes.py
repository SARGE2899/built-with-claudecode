"""
Gaussian Naive Bayes classification.

Concept: assumes each feature is normally distributed within each class,
and treats all features as independent given the class ("naive"). It
learns a mean and variance per feature per class, then classifies a new
point by asking "under which class's Gaussian bell curves is this point
more probable?" (combined with Bayes' rule and the class priors).

With exactly 1 training example per class, the "variance" of each
class's Gaussian is computed from a single point (0, before smoothing),
so sklearn's tiny var_smoothing epsilon is doing all the work of keeping
these distributions non-degenerate.
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = GaussianNB()
model.fit(X, y)

print("=== Naive Bayes internal state ===")
print(f"Classes: {model.classes_}")
print(f"Class priors (fraction of each class in training data): {model.class_prior_}")
print("\nPer-class per-feature mean (theta_):")
print(pd.DataFrame(model.theta_, index=model.classes_, columns=X.columns))
print("\nPer-class per-feature variance (var_), mostly just the smoothing "
      "epsilon since each class has only 1 sample:")
print(pd.DataFrame(model.var_, index=model.classes_, columns=X.columns))

print("\n=== Final predictions ===")
preds = model.predict(X)
probs = model.predict_proba(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=0)={probs[i][0]:.4f}  P(class=1)={probs[i][1]:.4f}")
