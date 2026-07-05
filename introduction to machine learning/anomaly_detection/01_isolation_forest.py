"""
Isolation Forest.

Concept: anomalies are "few and different", so they're easier to
isolate than normal points. Isolation Forest builds many random binary
trees that split on random features at random thresholds; a point that
can be separated from everything else in just a few splits (a short
average path length across all trees) is scored as anomalous, while
points buried deep in a dense cluster take many splits to isolate.

WHY THIS SCRIPT AUGMENTS THE DATA (read before running the rest of this
folder): anomaly detection needs a notion of "normal density" to define
what counts as abnormal -- with only the project's 2 real rows, there is
no density to measure at all, so every algorithm in this folder would
either crash or produce a meaningless answer. Per the project's own
2-row constraint, `data/loan_data.csv` itself is never touched. Instead,
this script derives a small synthetic neighborhood FROM the 2 real rows
themselves (see `augment_data()` below): a handful of points jittered
around each real row (standing in for "other applicants like this one"),
plus 2 deliberately extreme points representing genuine anomalies. Every
printed row is labeled with its real/synthetic source so it's never
ambiguous which 2 rows are the actual project data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["age", "income", "credit_score", "years_employed", "existing_debt"]


def augment_data():
    df = pd.read_csv("../data/loan_data.csv")
    real_a = df.iloc[0][FEATURE_COLS]  # approved=1 profile
    real_b = df.iloc[1][FEATURE_COLS]  # approved=0 profile

    # Small per-feature jitter (roughly the scale a "similar applicant"
    # might plausibly vary by), used only to generate synthetic
    # neighbors -- never to alter the 2 real rows themselves.
    rng = np.random.default_rng(0)
    jitter_scale = np.array([2, 5, 15, 1, 3])  # age, income, credit_score, years_employed, existing_debt

    def perturb(base_row, n):
        noise = rng.normal(loc=0.0, scale=jitter_scale, size=(n, len(FEATURE_COLS)))
        return np.round(base_row.values.astype(float) + noise).astype(int)

    synthetic_near_a = perturb(real_a, 8)
    synthetic_near_b = perturb(real_b, 8)

    # Deliberately extreme, implausible profiles -- not derived by small
    # jitter, but still expressed on the same feature axes as the real
    # rows, standing in for "genuinely anomalous applicant".
    synthetic_anomalies = np.array([
        [90, 200, 300, 45, 95],
        [19, 4, 310, 0, 90],
    ])

    rows = [real_a.values.astype(int), real_b.values.astype(int)]
    rows += list(synthetic_near_a) + list(synthetic_near_b) + list(synthetic_anomalies)
    sources = (
        ["real_A", "real_B"]
        + ["synthetic_near_A"] * len(synthetic_near_a)
        + ["synthetic_near_B"] * len(synthetic_near_b)
        + ["synthetic_anomaly"] * len(synthetic_anomalies)
    )
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    return X, sources


X, sources = augment_data()

print("=== Augmented dataset (2 REAL rows + synthetic neighbors/anomalies) ===")
print(X.assign(source=sources))

model = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
model.fit(X)

print("\n=== Isolation Forest internal state ===")
print(f"Number of trees: {len(model.estimators_)}")
print("contamination=0.1 tells the model to expect ~10% of these 20 rows "
      "(i.e. ~2) to be anomalies when choosing its score threshold -- "
      "which happens to exactly match our 2 deliberate synthetic_anomaly "
      "rows, by design of this demo.")

print("\n=== Final anomaly scores (higher = more normal; -1 = flagged anomaly) ===")
preds = model.predict(X)
scores = model.decision_function(X)
for source, pred, score in zip(sources, preds, scores):
    flag = "ANOMALY" if pred == -1 else "normal"
    print(f"  [{source:18s}] score={score:+.4f}  -> {flag}")

n_real_anomalies_caught = sum(
    1 for s, p in zip(sources, preds) if s == "synthetic_anomaly" and p == -1
)
print(f"\n{n_real_anomalies_caught}/2 deliberate synthetic anomalies were correctly "
      "flagged, isolated from the rest in fewer average tree splits.")
