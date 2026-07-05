"""
Repetition Penalty.

Concept: `01_greedy_decoding.py` showed a real failure mode -- once a
token becomes the argmax, greedy has no mechanism to ever stop repeating
it. Repetition penalty fixes this directly by penalizing the logits of
tokens that have ALREADY appeared in the generated sequence so far,
BEFORE picking the next token: for each token seen so far, divide its
logit by a penalty factor > 1 if positive (shrinking it toward zero) or
multiply by the penalty if negative (making it more negative) --
weakening its chance of being picked again without banning it outright.

This script uses a COUNT-based penalty (a token's penalty compounds with
how many times it has already appeared: divide by penalty^count), so a
token repeatedly chosen keeps getting progressively less attractive
rather than being penalized once and then staying at a fixed, possibly
still-dominant score. Same fixed hypothetical "a model that unhelpfully
wants to repeat itself" logits as `01_greedy_decoding.py`'s weakness
demo, extended with two modestly-positive alternative logits so a
realistic penalty value (1.3) is actually strong enough to change the
outcome (dividing a positive logit by a realistic penalty can shrink it
toward zero, but never past a competitor fixed at exactly zero).
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}

# Same "repetition-prone" scenario as 01_greedy_decoding.py's weakness
# demo, but with 'the' and 'on' given modest positive logits instead of
# 0 -- otherwise NO finite penalty could ever push 'mat' below a
# competitor stuck at exactly 0 (dividing a positive number by a
# penalty > 1 shrinks it toward, but never past, zero).
BASE_LOGITS = np.array([2.5, 0.3, -1.0, 0.4, 3.0])  # the, cat, sat, on, mat
PENALTY = 1.3
N_STEPS = 8


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def apply_repetition_penalty(logits, counts):
    penalized = logits.copy()
    for idx, count in counts.items():
        if count == 0:
            continue
        if penalized[idx] > 0:
            penalized[idx] = penalized[idx] / (PENALTY ** count)
        else:
            penalized[idx] = penalized[idx] * (PENALTY ** count)
    return penalized


print("=== Base (unpenalized) logits every step would start from ===")
for word, idx in VOCAB.items():
    print(f"  {word!r:>5s}: {BASE_LOGITS[idx]:+.2f}")

print("\n=== WITHOUT repetition penalty ===")
generated_no_penalty = ["mat"]  # seed token
for step in range(N_STEPS):
    next_word = ID_TO_WORD[int(np.argmax(BASE_LOGITS))]
    generated_no_penalty.append(next_word)
print(f"Generated: {generated_no_penalty}")
print("Exactly the same failure as 01_greedy_decoding.py: 'mat' is the "
      "single highest-logit token, so plain greedy picks it every single "
      "time, forever.")

print(f"\n=== WITH repetition penalty (penalty={PENALTY}, count-based) ===")
generated_with_penalty = ["mat"]
counts = {idx: 0 for idx in VOCAB.values()}
counts[VOCAB["mat"]] = 1  # the seed token counts as already generated

for step in range(N_STEPS):
    penalized_logits = apply_repetition_penalty(BASE_LOGITS, counts)
    next_id = int(np.argmax(penalized_logits))
    next_word = ID_TO_WORD[next_id]
    rounded = {w: round(float(penalized_logits[i]), 3) for w, i in VOCAB.items()}
    print(f"  step {step}: penalized logits={rounded}  -> picks {next_word!r}")
    generated_with_penalty.append(next_word)
    counts[next_id] += 1

print(f"\n=== Final output ===")
print(f"WITHOUT penalty: {generated_no_penalty}")
print(f"WITH penalty:    {generated_with_penalty}")

unique_no_penalty = len(set(generated_no_penalty))
unique_with_penalty = len(set(generated_with_penalty))
print(f"\nDistinct tokens used -- without penalty: {unique_no_penalty}, "
      f"with penalty: {unique_with_penalty}")
print("\nNOTE: the naive infinite 'mat mat mat mat...' loop IS fixed -- "
      "'the' now appears in every other slot, something plain greedy "
      "could never produce. But look closely at the penalized logits "
      "above: repetition penalty doesn't ban a token outright, it only "
      "shrinks its logit proportionally to how often it's already "
      "appeared, so once 'mat' has been penalized enough to fall below "
      "'the', and then 'the' gets penalized in turn and falls back below "
      "'mat' again, the two simply keep leapfrogging each other -- a "
      "2-token cycle instead of a 1-token one. This is a real, honest "
      "property of this penalty formula (not a flaw specific to this toy "
      "example): it reduces degenerate repetition but doesn't guarantee "
      "eliminating cyclical patterns altogether, which is exactly why "
      "production systems often combine it with complementary techniques "
      "like no-repeat-n-gram blocking (banning a whole PHRASE from "
      "recurring, not just a single token).")
