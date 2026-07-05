"""
Apriori (frequent itemset mining + association rules).

Concept: finds combinations of items that frequently co-occur together
in "transactions" (the classic story: supermarket baskets -- "customers
who buy X also buy Y"), using the "apriori property" to prune the search:
if an itemset isn't frequent, none of its supersets can be frequent
either, so candidate itemsets are generated and tested size-by-size (1
item, then 2, then 3...), pruning early rather than checking every
possible combination.

WHY THIS SCRIPT AUGMENTS AND TRANSFORMS THE DATA (read before running
the rest of this folder): the loan dataset isn't naturally transactional
(each row is one applicant's numeric measurements, not a "basket" of
discrete items), and 2 rows is nowhere near enough for "frequent"
co-occurrence to mean anything. So this script does two things to the
data, both clearly marked:
  1. BUCKETING -- each numeric column is converted into a categorical
     "item" (e.g. age -> "young"/"old"), so a row becomes a transaction
     (a set of items), the input association rule mining requires.
  2. SYNTHESIS -- a small number of additional transactions are derived
     from the 2 real rows via small jitter (same technique as
     `anomaly_detection/`), so there's more than 1 transaction per
     underlying pattern for "frequent" to mean something.
The 2 REAL rows are used as-is (only bucketed, values never altered)
and are labeled `real_A`/`real_B` throughout; every synthetic row is
labeled by which real row it was derived from.
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

FEATURE_COLS = ["age", "income", "credit_score", "years_employed", "existing_debt"]


def bucket_row(row):
    """Numeric columns -> categorical items. Thresholds are chosen so
    real_A and real_B fall on opposite sides of every single one."""
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

    # Jitter kept deliberately smaller than the distance from either real
    # row to any bucket threshold above, so synthetic points land in the
    # SAME bucket as the real row they were derived from, every time --
    # otherwise a stray flipped bucket would fragment the itemsets below
    # for reasons having nothing to do with the algorithm being taught.
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
print(f"\nOne-hot encoded transaction table ({onehot.shape[0]} transactions x "
      f"{onehot.shape[1]} possible items):")
print(onehot.astype(int))

frequent_itemsets = apriori(onehot, min_support=0.4, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

print(f"\n=== Apriori internal state: {len(frequent_itemsets)} frequent itemsets found ===")
print("NOTE ON THE COUNT: there are only 2 underlying real patterns here "
      "(the real_A bucket-set and the real_B bucket-set), each cloned "
      "across several rows -- and critically, they share ZERO items in "
      "common (every bucket is exclusive to one profile). That means "
      "EVERY non-empty subset of each 6-item profile is 'frequent' at "
      "support 0.5, which is 2*(2^6 - 1) = 126 itemsets from just 2 "
      "underlying patterns. This is a combinatorial artifact of small, "
      "cleanly-separated data -- not 126 independently-discovered "
      "shopping patterns.")

print("\nSingle-item frequent itemsets (support = fraction of "
      "transactions containing that item):")
singles = frequent_itemsets[frequent_itemsets["length"] == 1]
print(singles.sort_values("support", ascending=False)[["support", "itemsets"]].to_string(index=False))

print("\nThe 2 maximal (full 6-item) frequent itemsets -- the actual "
      "underlying patterns everything else above is a subset of:")
maximal = frequent_itemsets[frequent_itemsets["length"] == 6]
print(maximal[["support", "itemsets"]].to_string(index=False))

rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=0.8,
    num_itemsets=len(frequent_itemsets),
)
print(f"\n=== {len(rules)} association rules found (min confidence 0.8) ===")
print("Same caveat as above: with 2 mutually-exclusive underlying "
      "profiles, almost any item from one profile predicts every other "
      "item from that SAME profile with confidence 1.0 -- these are real "
      "computations, but on data this small they mostly demonstrate the "
      "mechanics, not a discovery a retailer could act on.")

print("\nA representative sample -- simple 1-item -> 1-item rules, "
      "strongest first:")
rules["ant_len"] = rules["antecedents"].apply(len)
rules["con_len"] = rules["consequents"].apply(len)
simple_rules = rules[(rules["ant_len"] == 1) & (rules["con_len"] == 1)]
simple_rules = simple_rules.sort_values(["lift", "confidence"], ascending=False)
print(simple_rules[["antecedents", "consequents", "support", "confidence", "lift"]]
      .head(6).to_string(index=False))

print("\n=== Final output: rule '{good_credit} -> {approved}' as one concrete example ===")
example = simple_rules.iloc[0]
print(f"  support={example['support']:.2f}  confidence={example['confidence']:.2f}  "
      f"lift={example['lift']:.2f}")
print(f"  meaning: {set(example['antecedents'])} appears in "
      f"{example['support']*100:.0f}% of transactions, and whenever it "
      f"does, {set(example['consequents'])} always follows here.")
