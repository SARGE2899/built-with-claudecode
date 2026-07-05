"""
Word-level Tokenization.

Concept: the very first step of any language model pipeline -- turning
raw text into a sequence of integer ids the model can actually compute
with. Word-level tokenization is the simplest scheme: split on
whitespace, assign each unique word an id, and a sentence becomes a
list of those ids. Every other script in this project builds on exactly
this scheme and this sentence, so the vocabulary and ids fixed here
(the=0, cat=1, sat=2, on=3, mat=4) are the ones every later script
assumes.

The weakness word-level tokenization has that later, more sophisticated
schemes fix: the vocabulary must contain every whole word a model will
ever see, and any word not seen during vocabulary-building (out-of-
vocabulary, OOV) has no id at all -- there's no way to represent it
except an <unk> placeholder, throwing away all its information.
"""

SENTENCE = "the cat sat on the mat"

words = SENTENCE.split(" ")
print("=== Word tokenization internal state ===")
print(f"Raw sentence: {SENTENCE!r}")
print(f"Split into words: {words}")

# Build the vocabulary in first-appearance order -- this fixes the
# canonical id assignment every other script in this project reuses.
vocab = {}
for w in words:
    if w not in vocab:
        vocab[w] = len(vocab)

print(f"\nVocabulary (word -> id), built in first-appearance order:")
for w, i in vocab.items():
    print(f"  {w!r}: {i}")
print(f"Vocabulary size: {len(vocab)}")

id_to_word = {i: w for w, i in vocab.items()}

print("\n=== Final output: encoded token ids ===")
token_ids = [vocab[w] for w in words]
print(f"'{SENTENCE}' -> {token_ids}")

decoded = " ".join(id_to_word[i] for i in token_ids)
print(f"Decoded back: {token_ids} -> '{decoded}'")
assert decoded == SENTENCE, "round-trip should reproduce the original sentence exactly"
print("(round-trip matches the original sentence exactly)")

print("\nNOTE: 'the' appears twice in the sentence but only once in the "
      "vocabulary -- word tokenization maps every occurrence of the same "
      "word to the same id, which is exactly why the sequence (length 6) "
      "is longer than the vocabulary (size 5).")

print("\nNOTE ON OOV: a word never seen while building the vocabulary "
      "(e.g. 'dog') has no id whatsoever under this scheme -- it would "
      "need to be mapped to a special <unk> id, silently discarding "
      "all information about what the word actually was. "
      "`03_byte_pair_encoding.py` shows the subword scheme real LLMs use "
      "instead, precisely to avoid this failure mode.")
