"""
Linear Regression.

Concept: fits a straight-line (technically hyperplane) relationship
y = w1*x1 + w2*x2 + ... + b between the features and a continuous
target, choosing the weights that minimize squared prediction error.

Here we predict `income` from the other columns (age, credit_score,
years_employed, existing_debt, approved). With 2 data points and 5
features, the system is underdetermined -- there are infinitely many
hyperplanes that pass through both points exactly. sklearn's solver
returns the minimum-norm solution among them, so residual error is
exactly 0, but the specific coefficients found are just one of
infinitely many equally-perfect fits, not a uniquely "correct" answer.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

model = LinearRegression()
model.fit(X, y)

print("=== Linear regression internal state ===")
print("Learned coefficients (weight per feature):")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.6f}")
print(f"Intercept: {model.intercept_:.6f}")
print("\nNOTE: with 2 points and 5 features the system is underdetermined "
      "(infinite exact-fit solutions exist); these are just the minimum-norm one.")

print("\n=== Final predictions ===")
preds = model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
