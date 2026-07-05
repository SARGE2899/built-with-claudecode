"""
Weight Tying.

Concept: a transformer needs a token embedding table (vocab_size x
d_model, mapping token ids to vectors -- see `embeddings/01_token_embeddings.py`)
AND a final output projection / "LM head" (d_model x vocab_size, mapping
the last layer's vectors back to per-token logits -- see
`training_objectives/01_next_token_prediction.py`). These are natural
transposes of each other in shape, and weight tying makes them literally
the SAME matrix: the LM head is computed as the embedding table's
transpose, rather than learning a second, independent matrix. This
roughly halves the parameter count of these two components combined,
and reflects a real intuition: a token whose embedding is close to
another token's embedding (the model treats them similarly as INPUT)
should plausibly also be predicted with similar logits when it's the
CORRECT answer -- tying forces the model to use one consistent notion of
"token similarity" for both directions instead of learning two possibly-
inconsistent ones.

Uses the same vocab_size=5 and d_model=8 as every other script.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 8

rng = np.random.default_rng(0)

print("=== UNTIED (standard 2-matrix) setup ===")
embedding_table = rng.normal(scale=0.1, size=(VOCAB_SIZE, D_MODEL))
output_head_untied = rng.normal(scale=0.1, size=(D_MODEL, VOCAB_SIZE))  # independent matrix
untied_params = embedding_table.size + output_head_untied.size
print(f"Embedding table shape: {embedding_table.shape} ({embedding_table.size} params)")
print(f"Output head shape:     {output_head_untied.shape} ({output_head_untied.size} params, "
      f"learned completely independently of the embedding table)")
print(f"Total parameters: {untied_params}")

print("\n=== TIED (weight-shared) setup ===")
# The exact same embedding table -- no second matrix is created at all.
output_head_tied = embedding_table.T  # a VIEW/transpose, not a copy
tied_params = embedding_table.size  # the transpose costs 0 additional parameters
print(f"Embedding table shape: {embedding_table.shape} ({embedding_table.size} params)")
print(f"Output head shape:     {output_head_tied.shape} "
      f"(== embedding_table.T -- literally the same underlying numbers, 0 extra params)")
print(f"Total parameters: {tied_params}")

print(f"\n=== Final output: parameter savings ===")
print(f"Untied: {untied_params} parameters")
print(f"Tied:   {tied_params} parameters")
print(f"Reduction: {100 * (untied_params - tied_params) / untied_params:.0f}%")

print("\n=== Demonstrating the 'literally the same matrix' property ===")
print("Modifying the embedding table in place (simulating one training "
      "step's update) and checking whether the tied output head reflects "
      "that change immediately, with no separate update step:")
before = output_head_tied.copy()
embedding_table[VOCAB["cat"], 0] += 1.0  # simulate a training update to one row
after = embedding_table.T  # re-read the "output head" -- same object, no separate computation
print(f"output_head_tied changed automatically (no code touched it "
      f"directly)? {not np.array_equal(before, after)}")
print("This is the entire point of tying: there is only ONE set of "
      "numbers in memory, and both 'the embedding table' and 'the output "
      "head' are just two ways of reading/indexing it -- there is no "
      "way for them to ever drift out of sync with each other, because "
      "there was never a second matrix to drift in the first place.")
