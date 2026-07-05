"""
Stacking.

Concept: trains several different base models (here: KNN and a shallow
Decision Tree), then trains a "meta-model" (Logistic Regression) whose
inputs are the base models' own predictions, learning how to best
combine them. To avoid the meta-model just memorizing the base models'
training answers, sklearn generates their input predictions via
cross-validation (holding out each fold while generating its prediction).

With only 2 samples, the smallest possible cross-validation is
"leave-one-out": each fold trains on 1 sample and predicts the other.
That means every training fold contains only ONE class -- sklearn
warns about this (folds aren't properly stratified) rather than erring
out, since KNN/DecisionTree can still technically fit and predict from
a single class. We use base learners that tolerate this; a solver like
plain LogisticRegression as a BASE learner would fail outright on a
single-class fold, which is exactly why it's used only as the final
meta-model here, trained on the full un-split data.
"""

import warnings

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

base_estimators = [
    ("knn", KNeighborsClassifier(n_neighbors=1)),
    ("dt", DecisionTreeClassifier(max_depth=1)),
]

model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(),
    cv=LeaveOneOut(),  # smallest possible CV split for only 2 samples
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model.fit(X, y)
    if caught:
        print("NOTE: sklearn warned that cross-validation folds aren't "
              "properly stratified. This is expected: with leave-one-out "
              "on 2 samples, each training fold has exactly 1 sample, so "
              "it can only ever contain 1 of the 2 classes. KNN and "
              "DecisionTree tolerate this fine (they just predict the one "
              "class they saw); this is why they were chosen as base "
              "learners instead of something like LogisticRegression, "
              "which would raise an error trying to fit on a single "
              "class.\n")

print("=== Stacking internal state ===")
print("Base estimators:", [name for name, _ in base_estimators])
print("\nMeta-model (final_estimator_) learned coefficients -- how much it "
      "trusts each base model's prediction:")
print(f"  coefficients: {model.final_estimator_.coef_}")
print(f"  intercept: {model.final_estimator_.intercept_}")

print("\nWhat each base model predicts standalone, for comparison:")
for name, est in model.named_estimators_.items():
    print(f"  {name}: {est.predict(X)}")

print("\n=== Final predictions (from the meta-model, combining both bases) ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")

if not (preds == y.values).all():
    knn_loo = cross_val_predict(KNeighborsClassifier(n_neighbors=1), X, y, cv=LeaveOneOut())
    print("\n=== Why this looks wrong, explained (not hidden) ===")
    print(f"Meta-model was TRAINED on leave-one-out predictions: {knn_loo} "
          f"paired with true labels {y.values}.")
    print("With only 2 rows, leave-one-out means each base model, while "
          "generating its training meta-feature for row i, was fit ONLY on "
          "the other row -- so it always predicts the OTHER row's class, "
          "which is the opposite of row i's own class in this 2-class, "
          "2-row dataset. The meta-model correctly learns that pattern: "
          "'when the base models say class 0, the answer is class 1' "
          "(a negative coefficient, printed above).")
    print("But standalone predictions above show the base models, refit "
          "on the FULL data for actual inference, correctly predict each "
          "row's own true label. Feeding these (now-correct) predictions "
          "into a meta-model trained to invert them produces exactly "
          "backwards final predictions. This is a real, instructive "
          "failure mode of stacking with too little data for cross-"
          "validation to be representative -- not a bug in this script.")
