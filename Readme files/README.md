# Readme files

Shared library of UI/visual assets used across READMEs in this repo. Not tied
to any single project folder — anything here can be reused by the root
README or any project's own README.

## Contents

- [`banners/built-with-claude-code/`](banners/built-with-claude-code/) —
  animated "Built with Claude Code" banner used at the top of the root
  [README.md](../README.md). See that folder's own README for the design
  rationale and how to regenerate it.

## Conventions

Each asset lives in its own folder with:
- `src/` — editable source (never just the exported file)
- a script to regenerate the final export
- the final static export (the only thing actually embedded in a README)
- a `README.md` explaining what it is and how to rebuild it

Brand reference used across these assets:

| Token | Value |
|---|---|
| Coral / terracotta (primary) | `#D97757` |
| Dark | `#141413` |
| Cream / off-white | `#FAF9F5` |
| Muted gray | `#B0AEA5` |
