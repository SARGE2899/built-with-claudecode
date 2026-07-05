"""
Beam Search.

Concept: greedy decoding (`01_greedy_decoding.py`) commits to the single
best-looking token at EVERY step, one at a time, and can never
reconsider -- but the token that looks best right now might lead to
worse options later, while a slightly-less-likely token now might open
up a much better continuation. Beam search hedges against this: instead
of keeping only the single best partial sequence, it keeps the top
`beam_width` partial sequences (by total cumulative log-probability) at
every step, expands EACH of them with every possible next token, and
again keeps only the overall best `beam_width` results -- so a
second-best first choice survives long enough for its own best
continuation to be discovered, instead of being discarded prematurely.

This script uses a hand-built 2-step scenario, specifically constructed
(not random/arbitrary) so that greedy's step-1 choice is NOT part of the
overall best 2-step sequence -- the textbook motivating case for beam
search. Sequence log-probabilities are compared directly against
`01_greedy_decoding.py`'s greedy choice.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def log_softmax(x):
    return np.log(softmax(x))


# Step 1 logits: 'mat' edges out 'on' as the single best next token
# (1.4 vs 1.3) -- greedy commits to 'mat' here and never looks back.
step1_logits = np.array([-1.0, -1.0, -1.0, 1.3, 1.4])  # the, cat, sat, on, mat

# Step 2 logits, DEPENDING on which token step 1 produced:
# - if step 1 was 'mat': no strong follow-up exists (best is a modest 0.3)
# - if step 1 was 'on': 'mat' is an overwhelmingly strong follow-up (4.0)
step2_logits_given = {
    "mat": np.array([0.3, 0.1, 0.0, 0.2, -2.0]),
    "on": np.array([-1.0, -1.0, -1.0, -2.0, 4.0]),
}

step1_logprobs = log_softmax(step1_logits)

print("=== Step 1 internal state ===")
for word, idx in VOCAB.items():
    print(f"  {word!r:>5s}: logit={step1_logits[idx]:+.2f}  logprob={step1_logprobs[idx]:+.4f}")

print("\n=== Greedy decoding's path (for comparison) ===")
greedy_first = ID_TO_WORD[int(np.argmax(step1_logits))]
greedy_step2_logprobs = log_softmax(step2_logits_given[greedy_first])
greedy_second = ID_TO_WORD[int(np.argmax(step2_logits_given[greedy_first]))]
greedy_total_logprob = step1_logprobs[VOCAB[greedy_first]] + greedy_step2_logprobs[VOCAB[greedy_second]]
print(f"  step 1: greedily picks {greedy_first!r} (highest single logit, 1.4 > 1.3) "
      "-- 'on' is discarded here and can never be reconsidered")
print(f"  step 2: greedily picks {greedy_second!r} given {greedy_first!r}")
print(f"  total sequence log-probability: {greedy_total_logprob:.4f}")

print("\n=== Beam search (beam_width=2) internal state ===")
BEAM_WIDTH = 2
# Keep the top BEAM_WIDTH candidates after step 1, not just the top 1.
top1_idx = np.argsort(-step1_logprobs)[:BEAM_WIDTH]
beams = [([ID_TO_WORD[i]], step1_logprobs[i]) for i in top1_idx]
print(f"Beams kept after step 1 (top {BEAM_WIDTH} by log-probability, not just top 1):")
for seq, logprob in beams:
    print(f"  {seq}  cumulative logprob={logprob:+.4f}")

print("\nExpanding EVERY kept beam with every possible step-2 token:")
candidates = []
for seq, logprob in beams:
    first_word = seq[0]
    step2_logprobs = log_softmax(step2_logits_given[first_word])
    for word, idx in VOCAB.items():
        total = logprob + step2_logprobs[idx]
        candidates.append((seq + [word], total))
        print(f"  {seq} + {word!r:>5s} -> cumulative logprob={total:+.4f}")

candidates.sort(key=lambda c: -c[1])
print(f"\nAll candidates, sorted by total log-probability, top {BEAM_WIDTH} kept:")
for seq, logprob in candidates[:BEAM_WIDTH]:
    print(f"  {seq}  logprob={logprob:+.4f}")

best_seq, best_logprob = candidates[0]

print(f"\n=== Final output: beam search's best sequence vs. greedy's ===")
print(f"Greedy:      {[greedy_first, greedy_second]}  logprob={greedy_total_logprob:.4f}")
print(f"Beam search: {best_seq}  logprob={best_logprob:.4f}")
assert best_logprob > greedy_total_logprob, "beam search should find a strictly better sequence here"
print(f"\nBeam search found a sequence with HIGHER total probability than "
      f"greedy ({np.exp(best_logprob):.4f} vs {np.exp(greedy_total_logprob):.4f}, "
      f"as raw probability) -- purely because it kept 'on' around as a "
      f"hypothesis after step 1 even though it wasn't the single best "
      f"immediate choice (1.3 logit vs. 'mat's 1.4). Greedy discarded 'on' "
      f"permanently at step 1 and never discovers the excellent 'on' -> "
      f"'mat' continuation (logit 4.0) hiding behind it.")
