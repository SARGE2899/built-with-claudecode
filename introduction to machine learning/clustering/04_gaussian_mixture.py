"""
Gaussian Mixture Model (GMM) Clustering.

Concept: like K-Means, but each cluster is modeled as a full Gaussian
distribution (with its own mean AND covariance/shape, not just a
centroid), and points get soft probabilities of belonging to each
cluster instead of a single hard assignment.

With 2 components and 2 points, each Gaussian ends up fit to exactly 1
point -- and the variance of a distribution estimated from a single
point is mathematically 0 (a distribution with zero spread), which
would normally be a singular covariance matrix. sklearn's
GaussianMixture already defaults reg_covar=1e-6 (a tiny constant added
to the diagonal) for exactly this situation, so it runs unmodified --
but the resulting covariances below are that artificial floor value,
not anything actually learned from data.
"""

import pandas as pd
from sklearn.mixture import GaussianMixture

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

model = GaussianMixture(n_components=2, random_state=0)  # reg_covar left at sklearn's default (1e-6)
model.fit(X)

print("=== Gaussian Mixture internal state ===")
print("Learned means (one per component):")
print(pd.DataFrame(model.means_, columns=X.columns))
print(f"\nConverged: {model.converged_}, iterations: {model.n_iter_}")
print("\nLearned covariances -- these equal exactly reg_covar (1e-6), "
      "sklearn's default floor, NOT a meaningfully-estimated spread, "
      "since each component saw only 1 training point:")
for i, cov in enumerate(model.covariances_):
    print(f"Component {i} covariance diagonal: {cov.diagonal()}")
print("\nNOTE: with true 0 regularization this would be a singular "
      "(all-zero) covariance matrix and GMM would raise an error -- a "
      "single point has no spread to measure. sklearn's default "
      "reg_covar=1e-6 is already the minimum needed to keep the model "
      "numerically valid; we didn't need to touch it.")

print("\n=== Final soft cluster assignments ===")
probs = model.predict_proba(X)
labels = model.predict(X)
for i in range(len(X)):
    print(f"  row {i} (age={X.iloc[i]['age']}) -> cluster {labels[i]}, "
          f"P(component 0)={probs[i][0]:.4f}, P(component 1)={probs[i][1]:.4f}")
