"""
Ridge and Lasso Regression.

Concept: both are Linear Regression with a penalty added for large
coefficients ("regularization"), which fights overfitting. Ridge (L2
penalty) shrinks all coefficients smoothly toward zero; Lasso (L1
penalty) can shrink some coefficients all the way to exactly zero,
effectively doing feature selection.

With only 2 points, plain Linear Regression already fits perfectly
(0 error) by exploiting its freedom from having more features than
samples -- so here regularization's visible effect is purely to pull
those coefficients back down at the cost of some fit error, which is
the clearest possible illustration of the bias-variance tradeoff these
methods exist for.
"""

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

plain = LinearRegression().fit(X, y)
ridge = Ridge(alpha=5.0).fit(X, y)
lasso = Lasso(alpha=5.0).fit(X, y)

print("=== Coefficient comparison: plain vs Ridge vs Lasso ===")
comparison = pd.DataFrame({
    "plain_linear": plain.coef_,
    "ridge (L2)": ridge.coef_,
    "lasso (L1)": lasso.coef_,
}, index=X.columns)
print(comparison)
print(f"\nIntercepts -- plain: {plain.intercept_:.4f}, "
      f"ridge: {ridge.intercept_:.4f}, lasso: {lasso.intercept_:.4f}")

print("\nNotice Lasso drives some coefficients to exactly 0 (feature "
      "selection); Ridge shrinks all of them but rarely to exactly 0.")

print("\n=== Final predictions ===")
for name, model in [("plain", plain), ("ridge", ridge), ("lasso", lasso)]:
    preds = model.predict(X)
    print(f"\n{name}:")
    for i in range(len(X)):
        print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
