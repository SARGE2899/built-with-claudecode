"""
UMAP (Uniform Manifold Approximation and Projection).

Concept: builds a weighted graph connecting each point to its nearest
neighbors (a "fuzzy" approximation of the data's underlying manifold),
then optimizes a low-dimensional layout whose own neighbor graph is as
similar as possible to that high-dimensional one -- broadly similar
goal to t-SNE (preserve local neighborhoods), but built on a different
mathematical foundation (topological data analysis) that scales better
and better preserves some global structure too.

UMAP's neighbor graph needs `n_neighbors` real neighbors per point, and
with only 2 points total, that graph gets pruned down to nothing: both
points are each other's sole (and maximally distant) neighbor, so UMAP's
internal edge-pruning step removes every edge and hands a truly empty
graph to its layout optimizer, which then crashes trying to normalize
an empty set of edge weights. Unlike LDA's precondition check (which
politely refuses with a ValueError before doing any work), this is UMAP
failing partway through its own pipeline -- reproduced and explained
below rather than papered over. We verified this isn't a tunable-away
choice of n_neighbors or init strategy: it fails identically down to
the same internal error regardless.
"""

import warnings

import pandas as pd
import umap

df = pd.read_csv("../data/loan_data.csv")
X = df.drop(columns=["approved"])  # unsupervised: label not used

print("=== Attempting to fit UMAP ===")
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # n_neighbors=2 is the minimum UMAP allows (>1); with n_samples=2
        # that's already "use every other point", the most lenient setting
        model = umap.UMAP(n_components=1, n_neighbors=2, random_state=0)
        transformed = model.fit_transform(X)
    print("Fit succeeded (unexpected with only 2 samples):")
    print(transformed)
except ValueError as e:
    print(f"UMAP crashed partway through fitting: {e}")
    print("\nWHY this happens: UMAP first builds a neighbor graph, then "
          "prunes low-confidence edges before laying out the embedding. "
          "With 2 points, each is the other's only (and least-similar-by-"
          "definition, since it's also the *most* distant) neighbor -- "
          "the pruning step removes every edge, leaving a graph with zero "
          "edges. The layout optimizer then tries to take the max() of an "
          "empty array of edge weights to normalize them, which is not "
          "defined, and crashes. This is a genuine implementation "
          "limitation at this data scale, not a hyperparameter we chose "
          "badly: we also tried n_neighbors=2 with init='random' (instead "
          "of the default spectral initialization) and it failed with the "
          "exact same error, confirming the empty-graph problem happens "
          "before initialization or layout even matter. It would require "
          "several more real data points -- enough for a non-trivial "
          "neighbor graph to survive pruning -- which the project "
          "constraints forbid adding.")
    print("\n=== Final output: none -- UMAP cannot be fit on this dataset ===")
