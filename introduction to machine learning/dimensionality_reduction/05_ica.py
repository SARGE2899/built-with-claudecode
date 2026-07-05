"""
Independent Component Analysis (ICA).

Concept: like PCA, but instead of maximizing variance, ICA looks for
axes along which the projected data is as STATISTICALLY INDEPENDENT
(and non-Gaussian) as possible -- the classic use case is unmixing
several independently-recorded signals (e.g. separating individual
voices from overlapping microphone recordings) back into their sources.

Estimating "statistical independence" is fundamentally a question about
distributions, which requires many samples to even ask. With exactly 2
points, there is no distribution to measure independence over -- just 2
individual values per feature. sklearn still requires n_components <=
min(n_samples, n_features) = 2 here and will return components, but
what it hands back is just some rotation that reconstructs the 2
points, not a genuine unmixing of independent sources.
"""

import pandas as pd
from sklearn.decomposition import FastICA

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

model = FastICA(n_components=2, random_state=0, whiten="unit-variance")
transformed = model.fit_transform(X)

print("=== ICA internal state ===")
print(f"Number of iterations until convergence: {model.n_iter_}")
print("Unmixing matrix (components_):")
print(pd.DataFrame(model.components_, columns=X.columns, index=["IC1", "IC2"]))
print("\nNOTE: 'independent components' is a statistical notion that needs "
      "a distribution of samples to estimate -- with only 2 points, this "
      "is just some rotation that maps the 2 points to 2D, not a "
      "meaningful separation of independent sources.")
print("\nNOTE: the unmixing coefficients above are enormous (1e11+). ICA's "
      "whitening step divides by the data's covariance, and with only 2 "
      "centered points that covariance is rank-1 (all the 'spread' sits "
      "along a single direction) -- dividing by its near-zero eigenvalue "
      "in the other direction blows the numbers up. The final transformed "
      "coordinates below still come out clean because that blow-up exactly "
      "cancels back out, but the intermediate matrix is a visible symptom "
      "of asking for 2 independent directions from data that only truly "
      "varies along 1.")

print("\n=== Final transformed coordinates (5 features -> 2 components) ===")
for i in range(len(X)):
    print(f"  row {i}: {transformed[i]}")
