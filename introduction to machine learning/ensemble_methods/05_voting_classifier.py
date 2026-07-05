"""
Voting Classifier.

Concept: trains several different model types independently on the same
full training data (no meta-model, no cross-validation -- unlike
Stacking), then combines their predictions by voting. "Hard" voting
takes a majority vote of predicted classes; "soft" voting averages
predicted class probabilities and picks the highest average.

Unlike `04_stacking.py`, there's no leave-one-out cross-validation step
here to break down with only 2 samples -- each base model just fits on
the full 2-row dataset directly, exactly as if run standalone. So this
is a cleaner illustration of "combine multiple models' opinions" without
the small-data cross-validation failure mode Stacking runs into.
"""

import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

base_estimators = [
    ("logreg", LogisticRegression()),
    ("knn", KNeighborsClassifier(n_neighbors=1)),
    ("dt", DecisionTreeClassifier(max_depth=1)),
]

model = VotingClassifier(estimators=base_estimators, voting="soft")
model.fit(X, y)

print("=== Voting Classifier internal state ===")
print("What each base model predicts/estimates standalone (fit on the "
      "full data, no cross-validation involved):")
for name, est in model.named_estimators_.items():
    probs = est.predict_proba(X)
    print(f"  {name}: predictions={est.predict(X)}  P(class=1)={probs[:, 1]}")

print("\n=== Final predictions (soft-vote average of all base models) ===")
preds = model.predict(X)
probs = model.predict_proba(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}  "
          f"P(class=0)={probs[i][0]:.4f}  P(class=1)={probs[i][1]:.4f}")
