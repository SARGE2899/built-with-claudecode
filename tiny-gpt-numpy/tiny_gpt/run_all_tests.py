"""Run every stage's self-test in order, stopping at the first failure.

This mirrors the rule we followed while building: a stage's gradient
check must pass before the next stage is trusted. `train.py` and
`generate.py` are excluded by default since they run full training loops
(slow, and not "tests" in the pass/fail sense) -- pass --all to include
them.
"""

import subprocess
import sys

STAGES = [
    ("Stage 1: tokenizer", "tokenizer.py"),
    ("Stage 2: embeddings", "embeddings.py"),
    ("Stage 3-4: attention forward + backward gradcheck", "attention.py"),
    ("Stage 5a: layernorm gradcheck", "layernorm.py"),
    ("Stage 5b: feedforward gradcheck", "feedforward.py"),
    ("Stage 5c: multi-head attention gradcheck", "mha.py"),
    ("Stage 5d: full block gradcheck", "block.py"),
    ("Stage 6: full model + loss gradcheck", "model.py"),
]

SLOW_STAGES = [
    ("Stage 7: training loop smoke test", "train.py"),
    ("Stage 8: train + generate", "generate.py"),
]


def run(name, script):
    print(f"\n{'=' * 60}\n{name}  ({script})\n{'=' * 60}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n>>> FAILED: {name} <<<")
        return False
    return True


if __name__ == "__main__":
    stages = STAGES + SLOW_STAGES if "--all" in sys.argv else STAGES

    for name, script in stages:
        if not run(name, script):
            sys.exit(1)

    print(f"\n{'=' * 60}\nALL {len(stages)} STAGES PASSED\n{'=' * 60}")
