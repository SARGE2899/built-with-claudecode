"""
Full Transformer Block.

Concept: assembles every piece from `attention/` and this folder into
the actual repeating unit a decoder-only LLM stacks N times: GPT-2/3
and most modern LLMs use "pre-norm" ordering (LayerNorm BEFORE each
sublayer, not after), because it empirically trains more stably at
depth than the original Transformer paper's "post-norm" ordering:

    x = x + CausalMultiHeadAttention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

Each sublayer's input is normalized independently, the sublayer computes
its update, and that update is added back via a residual connection --
so both LayerNorm calls sit strictly INSIDE the residual path, and the
running residual stream (x) itself is never directly normalized or
overwritten, only added to.

This single block is exactly what `tiny_gpt/block.py` (in this repo's
sibling `tiny gpt/` project) stacks N times to build a full model; here
it's isolated as one clearly-labeled, traceable computation instead of
being one piece of a bigger training loop.
"""

import numpy as np

VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
ID_TO_WORD = {i: w for w, i in VOCAB.items()}
D_MODEL = 8
D_FF = D_MODEL * 4
N_HEADS = 2
HEAD_DIM = D_MODEL // N_HEADS
SENTENCE = "the cat sat on the mat"
token_ids = [VOCAB[w] for w in SENTENCE.split(" ")]
SEQ_LEN = len(token_ids)


def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, None]
    dim = np.arange(d_model)[None, :]
    angle_rates = 1.0 / (10000 ** (2 * (dim // 2) / d_model))
    angles = position * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def relu(x):
    return np.maximum(0, x)


rng = np.random.default_rng(0)
token_embedding_table = rng.normal(scale=0.1, size=(len(VOCAB), D_MODEL))
x = token_embedding_table[token_ids] + sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)

# --- learned parameters (randomly initialized -- untrained) ---
ln1_gamma, ln1_beta = np.ones(D_MODEL), np.zeros(D_MODEL)
ln2_gamma, ln2_beta = np.ones(D_MODEL), np.zeros(D_MODEL)
W_q = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_k = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_v = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W_o = rng.normal(scale=0.3, size=(D_MODEL, D_MODEL))
W1 = rng.normal(scale=0.3, size=(D_MODEL, D_FF))
W2 = rng.normal(scale=0.3, size=(D_FF, D_MODEL))

causal_mask = np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)


def causal_multi_head_attention(x):
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    Q = Q.reshape(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    K = K.reshape(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    V = V.reshape(SEQ_LEN, N_HEADS, HEAD_DIM).transpose(1, 0, 2)
    head_outs = []
    for h in range(N_HEADS):
        scores = (Q[h] @ K[h].T) / np.sqrt(HEAD_DIM)
        scores = np.where(causal_mask, -np.inf, scores)
        head_outs.append(softmax(scores, axis=-1) @ V[h])
    concatenated = np.concatenate(head_outs, axis=-1)
    return concatenated @ W_o


def feedforward(x):
    return relu(x @ W1) @ W2


print("=== Input to the block (token + positional embeddings) ===")
print(f"Shape: {x.shape}")

print("\n=== Step 1: LayerNorm -> Causal Multi-Head Attention -> residual add ===")
normed1 = layer_norm(x, ln1_gamma, ln1_beta)
attn_out = causal_multi_head_attention(normed1)
x = x + attn_out
print(f"Post-attention residual stream shape: {x.shape}")
for pos, vec in enumerate(x):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): {np.round(vec, 3)}")

print("\n=== Step 2: LayerNorm -> Feedforward -> residual add ===")
normed2 = layer_norm(x, ln2_gamma, ln2_beta)
ffn_out = feedforward(normed2)
x = x + ffn_out

print(f"\n=== Final output: transformer block output (shape {x.shape}) ===")
for pos, vec in enumerate(x):
    print(f"  position {pos} ({ID_TO_WORD[token_ids[pos]]!r}): {np.round(vec, 3)}")

print("\nNOTE: the output has the EXACT same shape as the block's input "
      "(seq_len, d_model) -- this is what makes transformer blocks "
      "stackable: block 2 can take block 1's output as its own input "
      "with zero shape changes needed, all the way up to however many "
      "layers (N) the model uses. A real LLM repeats this exact block "
      "N times (GPT-3-scale models use N=96), then applies one final "
      "LayerNorm and the output projection shown in "
      "`efficiency_and_adaptation/03_weight_tying.py`.")
