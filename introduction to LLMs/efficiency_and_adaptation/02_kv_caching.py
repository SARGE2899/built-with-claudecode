"""
KV (Key/Value) Caching.

Concept: during autoregressive GENERATION (not training -- training
processes a whole known sequence at once, see `training_objectives/`),
tokens are produced one at a time, and each new token needs attention
over every token generated so far. Naively, that means recomputing
Key and Value projections for the ENTIRE prefix at every single step,
even though the K/V vectors for already-seen tokens never change (their
input embeddings are fixed, and with causal masking, earlier tokens
never attend to later ones, so nothing about them depends on what comes
next). KV caching simply stores each token's K and V vectors the first
time they're computed, and reuses them at every later step -- only the
newest token's K/V ever need to be computed fresh.

Processes the canonical sentence WORD BY WORD (unlike every other script
in this project, which processes it all at once) -- this is the one
concept that's inherently about incremental, step-by-step processing,
so it's the natural fit for the canonical sentence used exactly as-is.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]

rng = np.random.default_rng(0)
embedding_table = rng.normal(scale=0.1, size=(len(VOCAB), D_MODEL))
W_k = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_v = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))

print("=== WITHOUT KV caching: recompute K/V for the whole prefix, every step ===")
total_kv_computations_no_cache = 0
for t in range(1, len(token_ids) + 1):
    prefix_ids = token_ids[:t]
    X_prefix = embedding_table[prefix_ids]
    K_full = X_prefix @ W_k   # recomputed for ALL t tokens...
    V_full = X_prefix @ W_v   # ...even the t-1 we already computed last step
    total_kv_computations_no_cache += t
    words_so_far = [ID_TO_WORD[i] for i in prefix_ids]
    print(f"  step {t}: prefix={words_so_far}  "
          f"-> recomputed K,V for all {t} token(s) (K shape {K_full.shape})")

print(f"\nTotal K/V vector computations across all {len(token_ids)} steps "
      f"(no caching): {total_kv_computations_no_cache} "
      f"(1+2+3+...+{len(token_ids)}, a triangular number)")

print("\n=== WITH KV caching: compute each token's K/V exactly once ===")
k_cache = []  # list of (d_model,) vectors, grows by exactly 1 per step
v_cache = []
total_kv_computations_cached = 0

for t, tok_id in enumerate(token_ids):
    x_new = embedding_table[tok_id]           # only the NEW token's embedding
    k_new = x_new @ W_k                        # only 1 new K vector computed
    v_new = x_new @ W_v                        # only 1 new V vector computed
    k_cache.append(k_new)
    v_cache.append(v_new)
    total_kv_computations_cached += 1
    print(f"  step {t + 1}: new token={ID_TO_WORD[tok_id]!r}  "
          f"-> computed 1 new K,V pair, cache now holds {len(k_cache)} total")

print(f"\nTotal K/V vector computations across all {len(token_ids)} steps "
      f"(with caching): {total_kv_computations_cached} "
      f"(exactly 1 per step, however long the sequence gets)")

print("\n=== Final output: correctness check -- cached K/V match recomputed K/V ===")
K_cached_final = np.stack(k_cache)
V_cached_final = np.stack(v_cache)
K_recomputed_final = embedding_table[token_ids] @ W_k
V_recomputed_final = embedding_table[token_ids] @ W_v

k_matches = np.allclose(K_cached_final, K_recomputed_final)
v_matches = np.allclose(V_cached_final, V_recomputed_final)
print(f"Cached K matrix identical to a full from-scratch recomputation? {k_matches}")
print(f"Cached V matrix identical to a full from-scratch recomputation? {v_matches}")

savings = total_kv_computations_no_cache - total_kv_computations_cached
print(f"\nK/V computations saved by caching: {savings} "
      f"({total_kv_computations_no_cache} -> {total_kv_computations_cached}, "
      f"a {100 * savings / total_kv_computations_no_cache:.0f}% reduction)")
print(f"\nNOTE: caching changes ZERO numerical results (verified above) -- "
      f"it is a pure efficiency optimization. The saving also grows "
      f"quadratically with sequence length: for a sequence of length n, "
      f"no-cache does n(n+1)/2 total K/V computations vs. caching's "
      f"exactly n -- at this project's n=6 the difference is modest "
      f"(21 vs 6), but for a real LLM generating thousands of tokens, "
      f"recomputing the entire growing prefix at every single step would "
      f"be prohibitively slow.")
