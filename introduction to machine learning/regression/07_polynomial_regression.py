"""
Polynomial Regression.

Concept: still linear regression under the hood, but first expands the
input features with polynomial terms (squares, pairwise products) via
PolynomialFeatures, letting a "linear" model fit curved relationships in
the original feature space.

Plain linear regression on this dataset (`01_linear_regression.py`) was
already underdetermined (2 samples, 5 features -- infinitely many exact
fits). Expanding to degree=2 polynomial features balloons that to `n`
output features (printed below), amplifying the same situation rather
than fixing it. As before, sklearn's solver returns the minimum-norm
solution among all of them; residual error is exactly 0 (or numerically
indistinguishable from 0), which here is a sign of an underdetermined
system, not proof the model discovered a meaningful curved relationship.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("=== Polynomial feature expansion ===")
print(f"Original features: {len(X.columns)} -> expanded features: {X_poly.shape[1]}")
print(f"Expanded feature names: {list(poly.get_feature_names_out(X.columns))}")

model = LinearRegression()
model.fit(X_poly, y)

print("\n=== Polynomial regression internal state ===")
print(f"Number of learned coefficients: {len(model.coef_)} "
      f"(for only {len(X)} training samples)")
print(f"Intercept: {model.intercept_:.6f}")
print("\nNOTE: with far more features than samples, this is even more "
      "underdetermined than plain linear regression -- an exact "
      "(0-residual) fit here reflects that surplus of freedom, not "
      "genuine polynomial structure discovered in the data.")

print("\n=== Final predictions ===")
preds = model.predict(X_poly)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
