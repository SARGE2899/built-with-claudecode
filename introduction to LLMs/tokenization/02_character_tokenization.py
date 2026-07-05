"""
Character-level Tokenization.

Concept: instead of one id per whole word, assign one id per individual
character (including spaces). This fixes word tokenization's
out-of-vocabulary problem completely -- any string at all can be
represented, since the character vocabulary needs only as many entries
as there are distinct characters, not distinct words. The cost is a
much longer sequence for the same sentence, and the model must now
learn word structure itself from character sequences rather than
getting whole words "for free" as single tokens.
"""

SENTENCE = "the cat sat on the mat"

chars = list(SENTENCE)
print("=== Character tokenization internal state ===")
print(f"Raw sentence: {SENTENCE!r}")
print(f"Split into characters: {chars}")

vocab = {}
for c in chars:
    if c not in vocab:
        vocab[c] = len(vocab)

print(f"\nVocabulary (character -> id), built in first-appearance order:")
for c, i in vocab.items():
    label = "' ' (space)" if c == " " else repr(c)
    print(f"  {label}: {i}")
print(f"Vocabulary size: {len(vocab)}")

id_to_char = {i: c for c, i in vocab.items()}

print("\n=== Final output: encoded token ids ===")
token_ids = [vocab[c] for c in chars]
print(f"'{SENTENCE}' -> {token_ids}")

decoded = "".join(id_to_char[i] for i in token_ids)
print(f"Decoded back: {token_ids} -> '{decoded}'")
assert decoded == SENTENCE
print("(round-trip matches the original sentence exactly)")

print(f"\n=== Comparison with 01_word_tokenization.py ===")
print(f"Word tokenization:      6 tokens, vocabulary size 5")
print(f"Character tokenization: {len(token_ids)} tokens, vocabulary size {len(vocab)}")
print("\nNOTE: character tokenization can represent ANY string using this "
      "same tiny vocabulary (e.g. 'dog' = ['d','o','g'], all already in "
      "the alphabet above) -- there is no out-of-vocabulary problem. The "
      "cost is a sequence roughly 4x longer for the same sentence, and "
      "the model must now spend some of its capacity learning that "
      "'c'+'a'+'t' forms a meaningful unit at all, rather than receiving "
      "'cat' as one token with its own learned meaning from the start.")
