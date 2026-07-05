"""
Decision Tree Regression.

Concept: like a classification tree, but each leaf predicts the average
target value of the training points that land there (instead of a
class vote), and splits are chosen to minimize squared error rather
than impurity.

With 2 points, a single split perfectly isolates each point into its
own leaf, so the tree predicts each training income value exactly --
textbook overfitting, visible in one glance because the model is so
small.
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

print("=== Decision tree regressor internal state (learned split rules) ===")
print(export_text(model, feature_names=list(X.columns)))

print("Feature importances:")
for name, imp in zip(X.columns, model.feature_importances_):
    print(f"  {name}: {imp:.4f}")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
