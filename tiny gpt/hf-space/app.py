import os
import pickle
import sys

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tiny_gpt"))

from generate import generate  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pkl")

with open(CHECKPOINT_PATH, "rb") as f:
    checkpoint = pickle.load(f)

params = checkpoint["params"]
hyperparams = checkpoint["hyperparams"]

tok = CharTokenizer.__new__(CharTokenizer)
tok.stoi = checkpoint["stoi"]
tok.itos = checkpoint["itos"]
tok.vocab_size = checkpoint["vocab_size"]

app = FastAPI(title="Tiny GPT Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sarge2899.github.io"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    generated_text: str


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is missing or empty")

    prompt = prompt.lower()
    unsupported = sorted(set(prompt) - set(tok.stoi.keys()))
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                "prompt contains characters this tiny model was never trained on: "
                f"{unsupported!r}. Its vocabulary is limited to the characters in "
                "its small training corpus (lowercase letters, spaces, basic punctuation)."
            ),
        )

    rng = np.random.default_rng()
    text = generate(
        params, tok, prompt,
        max_new_tokens=60,
        block_size=hyperparams["block_size"],
        temperature=0.8,
        rng=rng,
    )
    return GenerateResponse(generated_text=text)


@app.get("/")
def health():
    return {"status": "ok"}
