# Introduction to LLMs

A learning project: every core concept behind modern decoder-only LLMs
(GPT-style, Claude-style) implemented in isolation, on the same tiny
canonical sentence, organized by category, so they can be traced by hand
and compared side by side. This is the conceptual counterpart to this
repo's [`tiny gpt/`](../tiny%20gpt/) project, which builds one complete
working model from scratch -- here, each mechanism (tokenization,
attention, decoding, efficiency tricks, ...) is pulled out and shown on
its own, self-contained, rather than being one piece of a bigger model.

## The canonical example

**The sentence "the cat sat on the mat"** is the *only* base example used
anywhere in this project -- every script that operates on real "data"
uses this exact sentence and the vocabulary it defines:

```
the=0, cat=1, sat=2, on=3, mat=4    (5 unique words, 6 total tokens)
```

This vocabulary is established in `tokenization/01_word_tokenization.py`
and reused by id in every later script. `d_model=8` (the embedding
dimension) and `n_heads=2` are likewise fixed once (in `embeddings/` and
`attention/` respectively) and reused everywhere they're relevant.

Every weight matrix in this project (embedding tables, attention
projections, feedforward layers, ...) is randomly initialized and
**never trained** -- no training loop exists anywhere here. The point is
to watch each mechanism's internal mechanics operate correctly on data
simple enough to trace by hand, not to produce a model that's actually
learned anything. Where a script's numbers need to look a specific way
to make a mechanism's behavior clearly visible (mainly in
`decoding_strategies/`, which starts from a plausible *output*
distribution rather than a trained model that could produce one), the
logits are explicitly hand-specified and labeled as hypothetical, never
presented as if they came from training.

## Setup

```
pip install numpy
```

(No other dependencies -- everything in this project is pure NumPy.)

Every script is self-contained and run from inside its own folder:

```
cd tokenization
python 01_word_tokenization.py
```

## Folder structure

```
introduction to LLMs/
├── tokenization/              text -> token ids
├── embeddings/                token ids -> vectors, + positional information
├── attention/                 the core mechanism: mixing information across positions
├── architecture/              the pieces around attention: LayerNorm, FFN, residuals, and the full block
├── training_objectives/       what a language model is actually trained to predict
├── decoding_strategies/       turning a trained model's output distribution into actual generated text
└── efficiency_and_adaptation/ making a trained model cheaper to run and adapt
```

## tokenization/ -- text -> token ids

| File | Concept |
|---|---|
| `01_word_tokenization.py` | Word-level tokenization (establishes the canonical vocabulary every other script reuses) |
| `02_character_tokenization.py` | Character-level tokenization |
| `03_byte_pair_encoding.py` | Byte Pair Encoding (BPE) |

## embeddings/ -- token ids -> vectors, + positional information

| File | Concept |
|---|---|
| `01_token_embeddings.py` | Token embedding lookup table |
| `02_positional_encoding.py` | Sinusoidal vs. learned positional encoding |
| `03_combining_embeddings.py` | Combining token + positional embeddings (the actual transformer input) |

## attention/ -- the core mechanism

| File | Concept |
|---|---|
| `01_scaled_dot_product_attention.py` | Scaled dot-product self-attention (Q, K, V) |
| `02_causal_masking.py` | Causal (autoregressive) masking |
| `03_multi_head_attention.py` | Multi-head attention |

## architecture/ -- the pieces around attention

| File | Concept |
|---|---|
| `01_layer_normalization.py` | Layer Normalization |
| `02_feedforward_and_residuals.py` | Position-wise feedforward network + residual connections |
| `03_full_transformer_block.py` | Full transformer block (pre-norm attention + FFN, assembled) |

## training_objectives/ -- what a language model is trained to predict

| File | Concept |
|---|---|
| `01_next_token_prediction.py` | Next-token prediction (autoregressive language modeling) |
| `02_teacher_forcing.py` | Teacher forcing vs. free-running generation |
| `03_masked_language_modeling.py` | Masked Language Modeling (BERT-style contrast) |

## decoding_strategies/ -- turning a distribution into generated text

| File | Concept |
|---|---|
| `01_greedy_decoding.py` | Greedy decoding |
| `02_temperature_sampling.py` | Temperature sampling |
| `03_top_k_sampling.py` | Top-k sampling |
| `04_top_p_nucleus_sampling.py` | Top-p (nucleus) sampling |
| `05_beam_search.py` | Beam search |
| `06_repetition_penalty.py` | Repetition penalty |

## efficiency_and_adaptation/ -- cheaper to run and adapt

| File | Concept |
|---|---|
| `01_lora.py` | LoRA (Low-Rank Adaptation) |
| `02_kv_caching.py` | KV (Key/Value) caching |
| `03_weight_tying.py` | Weight tying (shared input/output embedding matrix) |
| `04_quantization_concept.py` | Quantization (float32 -> int8/int4 precision/memory tradeoff) |

## A note on honesty about scale

A few scripts surface real, documented behaviors rather than idealized
ones: `01_greedy_decoding.py` and `06_repetition_penalty.py` show an
actual repetition failure mode and the real (imperfect) fix; `05_beam_search.py`
uses a deliberately constructed scenario where greedy decoding is
provably suboptimal, the textbook motivating case for beam search;
`training_objectives/02_teacher_forcing.py` shows a random model's
free-running generation genuinely collapse into a repeated token, rather
than a cleaned-up hypothetical. Nothing here is staged to look better
than the actual mechanism produces.
