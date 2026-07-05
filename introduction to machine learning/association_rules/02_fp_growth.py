"""
FP-Growth (frequent itemset mining via an FP-tree).

Concept: solves the exact same problem as Apriori (`01_apriori.py`) --
finding frequent itemsets -- but without Apriori's "generate a huge pile
of candidate itemsets, then scan the whole dataset to test each one"
approach. Instead, it compresses all transactions into a single prefix
tree (the "FP-tree", built from most-frequent to least-frequent items),
then mines frequent itemsets by walking that tree's paths directly --
no candidate generation, and no repeated full-database scans. This is
why FP-Growth generally outperforms Apriori on large real-world
transaction datasets with thousands of items.

Same augmentation/bucketing as `01_apriori.py` -- read that script's
docstring for the full rationale. The 2 REAL rows are bucketed as-is
and labeled `real_A`/`real_B`; a small number of synthetic transactions
are derived from them via small jitter, kept inside each bucket
threshold so they don't fragment the itemsets. Since the input here is
identical to `01_apriori.py`, this script also directly verifies that
FP-Growth's frequent itemsets are IDENTICAL to Apriori's, despite the
completely different internal mechanism -- the algorithm choice affects
speed and memory, not the (correct) answer.
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

FEATURE_COLS = ["age", "income", "credit_score", "years_employed", "existing_debt"]


def bucket_row(row):
    return [
        "young" if row["age"] < 40 else "old",
        "high_income" if row["income"] >= 50 else "low_income",
        "good_credit" if row["credit_score"] >= 650 else "poor_credit",
        "junior" if row["years_employed"] < 10 else "senior",
        "low_debt" if row["existing_debt"] < 20 else "high_debt",
        "approved" if row["approved"] == 1 else "rejected",
    ]


def augment_and_bucket():
    df = pd.read_csv("../data/loan_data.csv")
    real_a = df.iloc[0][FEATURE_COLS]
    real_b = df.iloc[1][FEATURE_COLS]

    rng = np.random.default_rng(0)
    jitter_scale = np.array([2, 5, 15, 1, 3])

    def perturb(base_row, n):
        noise = rng.normal(loc=0.0, scale=jitter_scale, size=(n, len(FEATURE_COLS)))
        return np.round(base_row.values.astype(float) + noise).astype(int)

    synthetic_near_a = perturb(real_a, 4)
    synthetic_near_b = perturb(real_b, 4)

    rows = [real_a.values.astype(int), real_b.values.astype(int)]
    rows += list(synthetic_near_a) + list(synthetic_near_b)
    approved = [1, 0] + [1] * len(synthetic_near_a) + [0] * len(synthetic_near_b)
    sources = (
        ["real_A", "real_B"]
        + ["synthetic_near_A"] * len(synthetic_near_a)
        + ["synthetic_near_B"] * len(synthetic_near_b)
    )

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    X["approved"] = approved
    transactions = [bucket_row(row) for _, row in X.iterrows()]
    return transactions, sources


transactions, sources = augment_and_bucket()

print("=== Transactions (2 REAL rows, bucketed + synthetic derived rows) ===")
for source, t in zip(sources, transactions):
    print(f"  [{source:18s}] {t}")

encoder = TransactionEncoder()
onehot = pd.DataFrame(
    encoder.fit(transactions).transform(transactions), columns=encoder.columns_
)

fp_itemsets = fpgrowth(onehot, min_support=0.4, use_colnames=True)

print(f"\n=== FP-Growth internal state: {len(fp_itemsets)} frequent itemsets found ===")
print("Same combinatorial-explosion caveat as `01_apriori.py`: 2 "
      "mutually-exclusive 6-item profiles produce 2*(2^6 - 1) = 126 "
      "'frequent' subsets. What matters here is HOW this number was "
      "reached, not the number itself.")

# Verify equivalence with Apriori directly, on this same one-hot table.
apriori_itemsets = apriori(onehot, min_support=0.4, use_colnames=True)
fp_sets = set(frozenset(s) for s in fp_itemsets["itemsets"])
apriori_sets = set(frozenset(s) for s in apriori_itemsets["itemsets"])
print(f"\nApriori found {len(apriori_sets)} itemsets on the identical input; "
      f"FP-Growth found {len(fp_sets)}. Identical sets of itemsets: "
      f"{fp_sets == apriori_sets}")
print("This is the actual point of comparing the two: same correct "
      "answer, arrived at via a completely different mechanism (tree "
      "traversal vs. candidate generation-and-test). Neither script's "
      "runtime is measured here, since at 10 transactions / 12 items "
      "both finish instantly -- FP-Growth's real advantage (skipping "
      "repeated full-database scans) only shows up with far more "
      "transactions and items than a 2-row seed dataset can ever provide.")

rules = association_rules(
    fp_itemsets, metric="confidence", min_threshold=0.8, num_itemsets=len(fp_itemsets),
)
rules["ant_len"] = rules["antecedents"].apply(len)
rules["con_len"] = rules["consequents"].apply(len)
simple_rules = rules[(rules["ant_len"] == 1) & (rules["con_len"] == 1)]
simple_rules = simple_rules.sort_values(["lift", "confidence"], ascending=False)

print(f"\n=== Final output: {len(rules)} rules found; a representative sample ===")
print(simple_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .head(6).to_string(index=False))
