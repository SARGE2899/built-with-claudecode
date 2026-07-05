---
title: Tiny GPT Demo
emoji: 🤖
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Tiny GPT Demo

FastAPI wrapper around [`tiny_gpt`](https://github.com/SARGE2899/built-with-claudecode/tree/main/tiny%20gpt) —
a GPT-style transformer built from scratch in pure NumPy (no PyTorch, no autograd).

The included `checkpoint.pkl` was trained on a small sample corpus (see
`train_checkpoint.py`) — it overfits that corpus, so output at low temperature
closely echoes the training text, which is expected for a tiny model on a tiny
corpus.

## API

`POST /generate`

Request:
```json
{ "prompt": "the dog" }
```

Response:
```json
{ "generated_text": "the dog barks at the fox..." }
```

Prompts are lowercased and must only contain characters from the training
corpus's vocabulary (lowercase letters, spaces, basic punctuation) — anything
else returns a 400.
