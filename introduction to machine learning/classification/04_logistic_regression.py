"""
Logistic Regression classification.

Concept: learns a weight for each feature and a bias, combines them
linearly (w . x + b), and squashes the result through a sigmoid function
to get a probability of the positive class. Weights are fit by
maximizing the likelihood of the observed labels.

With only 2 perfectly-separable points, this is a "perfect separation"
scenario: the optimizer can keep increasing the coefficients forever to
push predicted probabilities closer to 0 and 1, so it will not fully
converge in the mathematical sense. We cap max_iter and note the
resulting ConvergenceWarning instead of hiding it.
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = LogisticRegression(max_iter=1000)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        print("NOTE: solver did not fully converge. This is expected: with only "
              "2 perfectly separable points, logistic regression's coefficients "
              "can grow toward infinity to push probabilities toward exactly 0/1, "
              "so there is no finite optimum to converge to.\n")

print("=== Logistic regression internal state ===")
print("Learned coefficients (weight per feature):")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"  {name}: {coef:.6f}")
print(f"Intercept (bias): {model.intercept_[0]:.6f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
probs = model.predict_proba(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=1)={probs[i][1]:.6f}")
