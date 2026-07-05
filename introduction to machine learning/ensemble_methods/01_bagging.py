"""
Bagging (Bootstrap Aggregating).

Concept: trains many copies of the same base model, each on a random
bootstrap resample (sampling with replacement) of the training data,
then averages/votes their predictions. This is the general "combine
multiple models" recipe that Random Forest specializes (Random Forest =
bagging + decision trees + random feature subsets at each split).

With only 2 rows, every bootstrap resample can only contain some
combination of those same 2 points (possibly the same one twice), so
there's very little diversity between the base estimators to average
over -- the "wisdom of crowds" effect bagging relies on needs more raw
material than this to show up clearly.
"""

import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=10, random_state=0,
)
model.fit(X, y)

print("=== Bagging internal state ===")
print(f"Number of base estimators: {len(model.estimators_)}")
print("\nWhat training rows each bootstrap resample actually contained "
      "(by index), and what that tree predicts for row 0:")
for i, (tree, sample_idx) in enumerate(zip(model.estimators_, model.estimators_samples_)):
    pred = tree.predict(X.iloc[[0]])[0]
    print(f"  estimator {i}: resample used rows {list(sample_idx)} -> predicts {pred}")

print("\n=== Final predictions (majority vote across all base estimators) ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true={y.iloc[i]}  predicted={preds[i]}")
