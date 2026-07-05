"""
Teacher Forcing.

Concept: during TRAINING, the input fed into the model at every position
is always the TRUE previous token from the real training sentence,
regardless of what the model itself would have predicted there. This is
"teacher forcing" -- a teacher (the ground-truth data) forces the
correct answer as input at every step, so one wrong prediction at
position i can never corrupt the input the model sees at position i+1;
every position's loss is computed independently and in parallel (exactly
as in `01_next_token_prediction.py`).

This is NOT how the model operates at GENERATION time: there, there is
no ground truth to fall back on, so the model must feed its OWN
(possibly wrong) prediction back in as the next input -- a mismatch
between training and generation conditions sometimes called "exposure
bias". This script makes that contrast concrete by simulating a model
that gets one early prediction wrong, and tracing what happens to the
REST of the sequence under each regime.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
D_MODEL = 8
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
inputs = token_ids[:-1]
targets = token_ids[1:]
SEQ_LEN = len(inputs)

rng = np.random.default_rng(0)
embedding_table = rng.normal(scale=0.1, size=(VOCAB_SIZE, D_MODEL))
lm_head = rng.normal(scale=0.1, size=(D_MODEL, VOCAB_SIZE))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def predict_next(token_id):
    """Stand-in 'model': embed one token, project to vocab logits, return argmax id."""
    logits = embedding_table[token_id] @ lm_head
    probs = softmax(logits)
    return int(np.argmax(probs)), probs


print("=== TEACHER FORCING (how training actually works) ===")
print("At every position, the NEXT input is the true token from the "
      "sentence, no matter what the model predicted:\n")
for pos in range(SEQ_LEN):
    predicted_id, probs = predict_next(inputs[pos])
    correct = "correct" if predicted_id == targets[pos] else "WRONG"
    print(f"  position {pos}: input={ID_TO_WORD[inputs[pos]]!r:>6s}  "
          f"model predicts={ID_TO_WORD[predicted_id]!r:>6s} ({correct})  "
          f"true next={ID_TO_WORD[targets[pos]]!r:>6s}  "
          f"-> NEXT INPUT USED = {ID_TO_WORD[targets[pos]]!r} (the true token, always)")

print("\nNOTE: even where the model's prediction was wrong, the input fed "
      "at the next position was still the TRUE token from the sentence -- "
      "one bad prediction has zero effect on any other position's input. "
      "This is exactly why all 5 positions' losses in "
      "`01_next_token_prediction.py` can be computed simultaneously in "
      "one forward pass: none of them actually depend on each other's "
      "predictions, only on the fixed, real sentence.")

print("\n=== FREE-RUNNING / AUTOREGRESSIVE (how GENERATION actually works) ===")
print("Starting from just the first token, each next input is now the "
      "MODEL'S OWN previous prediction, not the true sentence:\n")
current_id = inputs[0]
generated = [current_id]
for pos in range(SEQ_LEN):
    predicted_id, probs = predict_next(current_id)
    matches_true_sentence = (
        pos + 1 < len(token_ids) and predicted_id == token_ids[pos + 1]
    )
    tag = "matches true sentence" if matches_true_sentence else "DIVERGED from true sentence"
    print(f"  step {pos}: fed in {ID_TO_WORD[current_id]!r:>6s}  -> "
          f"model predicts {ID_TO_WORD[predicted_id]!r:>6s}  ({tag})")
    generated.append(predicted_id)
    current_id = predicted_id  # <-- the model's OWN output becomes the next input

print(f"\n=== Final output: generated sequence vs. the true sentence ===")
print(f"True sentence:      {[ID_TO_WORD[i] for i in token_ids]}")
print(f"Free-running output: {[ID_TO_WORD[i] for i in generated]}")

print("\nNOTE: this untrained model's first prediction is already 'wrong' "
      "relative to the true sentence, and because free-running feeds that "
      "wrong prediction back in as the next input, EVERY subsequent step "
      "compounds on top of an already-diverged path -- contrast this with "
      "teacher forcing above, where each position's input was completely "
      "insulated from every other position's mistakes. This training/"
      "generation mismatch is exactly why real training pipelines "
      "sometimes mix in a bit of the model's own predictions during "
      "training (scheduled sampling) to reduce the gap.")
