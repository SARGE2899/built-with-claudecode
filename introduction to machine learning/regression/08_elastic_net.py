"""
Elastic Net Regression.

Concept: linear regression with BOTH an L1 (Lasso) and L2 (Ridge)
penalty at once, combined via `l1_ratio` (0 = pure Ridge, 1 = pure
Lasso, in between = a blend of both). It exists because pure Lasso can
behave erratically when features are correlated -- Elastic Net's L2
component stabilizes it while still keeping Lasso's ability to zero out
coefficients.

Same underlying story as `02_ridge_lasso.py`: with 2 points and 5
features, plain linear regression already fits exactly (0 error), so
regularization's only visible effect here is pulling coefficients down
at the cost of some fit error -- shown below across a few l1_ratio
values to make the Ridge<->Lasso blend directly visible.
"""

import pandas as pd
from sklearn.linear_model import ElasticNet

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["income"])
y = df["income"]

l1_ratios = [0.1, 0.5, 0.9]
models = {ratio: ElasticNet(alpha=5.0, l1_ratio=ratio).fit(X, y) for ratio in l1_ratios}

print("=== Elastic Net internal state: coefficients across l1_ratio ===")
comparison = pd.DataFrame(
    {f"l1_ratio={ratio}": model.coef_ for ratio, model in models.items()},
    index=X.columns,
)
print(comparison)
print("\nAs l1_ratio increases toward 1 (more Lasso-like), more "
      "coefficients get pulled toward/to exactly 0; toward 0 (more "
      "Ridge-like), shrinkage is smoother and spread across all of them.")

print("\n=== Final predictions (l1_ratio=0.5, an even Ridge/Lasso blend) ===")
mid_model = models[0.5]
preds = mid_model.predict(X)
for i in range(len(X)):
    print(f"  row {i}: true income={y.iloc[i]}  predicted={preds[i]:.4f}")
