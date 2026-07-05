"""
Byte Pair Encoding (BPE).

Concept: the subword scheme real LLMs actually use, striking a middle
ground between word tokenization (whole words, but OOV words have no
representation) and character tokenization (any string representable,
but long sequences with no whole-word tokens). BPE starts from
individual characters and repeatedly merges the single most frequent
ADJACENT PAIR of symbols across the whole training corpus into one new
symbol, building up a vocabulary of subwords: common whole words tend to
merge all the way into a single token, while rare words stay split into
smaller reusable pieces -- and any string can still always be represented
by falling back to individual characters when no larger merge applies.

Trained on a corpus this tiny (5 distinct words), BPE's behavior is
still genuine and instructive: it's implemented as the real algorithm
(count adjacent pairs, merge the most frequent, repeat), just run on a
small enough corpus to trace by hand. Ties are broken lexicographically
for determinism (a real implementation's exact tie-break doesn't matter
for correctness, only for reproducibility).
"""

from collections import Counter

SENTENCE = "the cat sat on the mat"
END = "</w>"  # marks a word boundary, so BPE never merges across word edges

words = SENTENCE.split(" ")
word_counts = Counter(words)

print("=== BPE internal state ===")
print(f"Corpus word counts: {dict(word_counts)}")

# Represent each distinct word as a tuple of symbols (initially single
# characters + an end-of-word marker), the classic BPE starting point.
word_symbols = {w: tuple(list(w) + [END]) for w in word_counts}
print("\nInitial per-word symbol sequences (character-level + end marker):")
for w, symbols in word_symbols.items():
    print(f"  {w!r} (x{word_counts[w]}): {symbols}")


def get_pair_counts(word_symbols, word_counts):
    pair_counts = Counter()
    for w, symbols in word_symbols.items():
        freq = word_counts[w]
        for a, b in zip(symbols, symbols[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts


def merge_pair(word_symbols, pair):
    merged = "".join(pair)
    new_word_symbols = {}
    for w, symbols in word_symbols.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_word_symbols[w] = tuple(new_symbols)
    return new_word_symbols


print("\n=== Merge steps ===")
merges = []
while True:
    pair_counts = get_pair_counts(word_symbols, word_counts)
    if not pair_counts:
        break
    max_count = max(pair_counts.values())
    if max_count <= 1:
        print(f"Stopping: no pair occurs more than once anymore "
              f"(remaining pairs: {dict(pair_counts)})")
        break
    # deterministic tie-break: lexicographically smallest pair among ties
    best_pair = min(p for p, c in pair_counts.items() if c == max_count)
    word_symbols = merge_pair(word_symbols, best_pair)
    merges.append(best_pair)
    print(f"  merge {len(merges)}: {best_pair} -> {''.join(best_pair)!r}  "
          f"(count={max_count})")
    print(f"    words now: {{ {', '.join(f'{w!r}: {s}' for w, s in word_symbols.items())} }}")

print(f"\n=== Final output: learned subword vocabulary after {len(merges)} merges ===")
final_symbols = sorted({s for symbols in word_symbols.values() for s in symbols})
print(f"Subword vocabulary: {final_symbols}")
print(f"Vocabulary size: {len(final_symbols)} (started from "
      f"{len({c for w in word_counts for c in w} | {END})} characters+marker)")

print("\nHow each word ends up tokenized:")
for w, symbols in word_symbols.items():
    print(f"  {w!r} -> {symbols}")

print("\nNOTE: 'the' (frequency 2, the most common word) merged all the "
      "way down to a single whole-word token, while 'cat'/'sat'/'mat' "
      "each stopped at 2 tokens sharing the merged subword 'at</w>' -- "
      "BPE discovered that shared root on its own, from co-occurrence "
      "statistics, without being told 'at' was meaningful. 'on' never "
      "merged at all: it shares no repeated adjacent pair with any other "
      "word in this tiny corpus, so it stays fully spelled out in "
      "characters -- exactly the graceful fallback that gives BPE no "
      "true out-of-vocabulary failure mode.")
