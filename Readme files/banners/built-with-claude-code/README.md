# Built with Claude Code — banner

<p align="center">
  <img src="output.webp" alt="Built with Claude Code" width="600">
</p>

## Concept

Three directions were considered before building:

1. **Terminal window with typing text** — the obvious default. Skipped: it's the
   cliché every "built with an AI coding tool" badge reaches for, and it reads as
   a technical demo rather than a brand moment.
2. **Particle network assembling into the wordmark** — visually rich, but at
   README width (~600px) and through GIF/WebP compression, particle-formed
   letterforms tend to read as noisy/illegible rather than crisp.
3. **Ambient gradient field + kinetic typography** *(chosen)* — a slow-drifting
   coral aurora behind staggered, blur-reveal typography, finished with a
   hand-choreographed accent line and a soft light sweep. This scales down
   cleanly, stays legible at small sizes, and reads as a deliberate brand
   moment rather than a tech demo.

## Choreography (6s seamless loop, 30fps)

| Time | Beat |
|---|---|
| 0.00s | Rest state: background glow at loop phase 0, type fully hidden |
| 0.30–1.55s | "Built", "with", "Claude Code" reveal in staggered sequence (blur + rise + slight overshoot settle) |
| 1.55–2.15s | Coral underline draws in beneath "Claude Code", traveling highlight dot at the tip |
| 2.55–3.55s | Soft diagonal light sweep crosses the accent word once |
| 4.30–4.85s | Underline retracts |
| 4.50–5.30s | Type fades out together (unified exit, not staggered) |
| 5.30–6.00s | Background-only tail — masks the loop seam |

Every property is a pure function of time `t` (see `src/animation.js`), and all
periodic motion (background drift, glow breathing) uses periods that evenly
divide the 6s loop, so frame 179 flows into frame 0 with no visible jump cut.

## Why WebP, and why lossy (not lossless)

The background is a soft multi-stop radial gradient; GIF's 256-color-per-frame
palette banded and dithered it badly (and ballooned to 20MB+ at this
resolution). Animated WebP keeps full 24-bit color, loops natively, and
renders inline on GitHub.

True lossless WebP and APNG were both tested directly (not assumed): at
1400×350/30fps/6s they land at **~35MB and ~52MB** respectively — 15-25x past
a reasonable README weight budget. So the deliverable is lossy WebP, tuned to
minimize the one artifact lossy WebP is prone to: chroma subsampling softens
saturated edges (like white/coral text on a dark background) because lossy
WebP always encodes chroma at 4:2:0, by design of the codec — no ffmpeg flag
turns that off for lossy mode. Two changes claw most of that back:

- **`-preset text`** — libwebp's encoder presets bias the rate-distortion
  search toward different content; the default preset is tuned for photos,
  the `text` preset (`5`) is tuned for high-contrast type/line-art, which is
  exactly what this banner is. Same bitrate, visibly less edge softening.
- **Dropped the grain/noise overlay** — a subtle film-grain texture was in an
  earlier pass. High-frequency noise is expensive for any lossy codec to
  hold onto, so it was quietly eating bits that would otherwise go toward
  text-edge fidelity. Removed; the gradient reads cleaner without it anyway.

Net result: **~2.0MB**, comfortably inside a 2-3MB budget, with noticeably
crisper type than the first pass at a similar bitrate.

## Verifying the encode

This repo's `ffmpeg` build can mux the animated WebP but can't reliably
demux/decode it back (a version-specific limitation, not a validity issue).
To confirm the file actually decodes and animates correctly, load it in a
real browser engine instead:

```js
// quick sanity check — navigate Chromium directly at the file and screenshot
const browser = await puppeteer.launch({ headless: 'new' });
const page = await browser.newPage();
await page.goto('file:///' + /* absolute path to output.webp, URI-encoded */);
await page.screenshot({ path: 'check.png' });
```

This is also the more meaningful test: GitHub renders README images through a
browser image pipeline, not through `ffmpeg`.

## Stack

- Plain HTML/CSS + [GSAP](https://gsap.com/) (used only for its easing
  functions — `expo.out`, `power2.in/inOut`, `back.out` — applied manually
  per-frame, not as a live timeline)
- [Puppeteer](https://pptr.dev/) captures a deterministic PNG frame sequence
  by calling `window.renderAtTime(t)` for each frame, so the render is
  frame-accurate regardless of machine speed
- `ffmpeg` (via `ffmpeg-static`, no system install required) encodes the PNG
  sequence into the final animated WebP

## Regenerating

```
cd "Readme files/banners/built-with-claude-code"
npm install
node render.js
```

This installs Puppeteer + ffmpeg-static + GSAP + the Inter font locally
(dev dependencies only, gitignored), captures 180 frames at 1400×350 — about
2.3x the README's `width="600"` display size, for crisp retina rendering —
and writes `output.webp` (~2.0MB).

To tweak timing or design, edit `src/animation.js` (timing/easing/motion) or
`src/index.html` (layout, colors, type). Open `src/index.html` directly in a
browser for a live real-time preview before re-rendering — it runs the exact
same `renderAtTime()` function driven by `requestAnimationFrame` instead of
Puppeteer's deterministic clock.

## Files

```
src/
  index.html       — markup, styles, brand colors, SVG underline
  animation.js     — deterministic renderAtTime(t); all motion/easing math
render.js          — Puppeteer capture + ffmpeg encode → output.webp
output.webp        — final deliverable, embedded in the repo README
```
