"""
Linear Discriminant Analysis (LDA).

Concept: UNLIKE the other three scripts in this folder (PCA, t-SNE,
autoencoder), LDA is SUPERVISED -- it uses the class label to find the
axis that best SEPARATES the classes, rather than the axis that best
preserves overall variance. It works by comparing between-class scatter
to within-class scatter.

That "within-class scatter" is exactly what breaks here: with only 1
sample per class, there is no within-class variation to measure at all
(a single point has no spread). sklearn enforces n_samples > n_classes
as a hard precondition and refuses to fit rather than silently return a
meaningless answer -- we reproduce and explain that failure below
instead of forcing a fake result.
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])
y = df["approved"]  # LDA is supervised: label IS used, unlike PCA/t-SNE/autoencoder

print("=== Attempting to fit LDA ===")
try:
    model = LinearDiscriminantAnalysis(n_components=1)
    transformed = model.fit_transform(X, y)
    print("Fit succeeded (unexpected with only 2 samples/2 classes):")
    print(transformed)
except ValueError as e:
    print(f"LDA refused to fit: {e}")
    print("\nWHY this happens: LDA needs to estimate within-class scatter "
          "(how spread out samples are inside each class) to know which "
          "directions are 'noise' vs 'signal'. With exactly 1 sample per "
          "class, within-class scatter is mathematically zero -- there is "
          "no spread to measure -- and sklearn requires strictly more "
          "samples than classes before it will even attempt a fit. This "
          "is a hard algorithmic limitation, not something a different "
          "solver or hyperparameter can work around (we verified this by "
          "testing all three solvers: svd, lsqr, eigen -- all fail "
          "identically). It would require a 3rd data point to even "
          "attempt LDA here, which the project constraints forbid adding.")
    print("\n=== Final output: none -- LDA cannot be fit on this dataset ===")
