// Renders the "Built with Claude Code" banner animation to a static
// animated image (WebP) by:
//   1. Loading src/index.html in headless Chromium (Puppeteer)
//   2. Stepping a deterministic clock (window.renderAtTime(t)) frame by
//      frame and screenshotting the stage element
//   3. Encoding the PNG frame sequence into an animated WebP with ffmpeg
//
// Usage: node render.js

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const puppeteer = require('puppeteer');
const ffmpegPath = require('ffmpeg-static');

const FPS = 30;
const DURATION = 6.0;
const TOTAL_FRAMES = Math.round(FPS * DURATION);

const ROOT = __dirname;
const FRAMES_DIR = path.join(ROOT, 'frames');
const SRC_HTML = 'file://' + path.join(ROOT, 'src', 'index.html');
const OUT_WEBP = path.join(ROOT, 'output.webp');

async function main() {
  if (fs.existsSync(FRAMES_DIR)) fs.rmSync(FRAMES_DIR, { recursive: true, force: true });
  fs.mkdirSync(FRAMES_DIR, { recursive: true });

  console.log(`Launching headless Chromium...`);
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--force-color-profile=srgb', '--font-render-hinting=none'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 350, deviceScaleFactor: 1 });
  await page.evaluateOnNewDocument(() => {
    window.__CAPTURE_MODE__ = true;
  });
  await page.goto(SRC_HTML, { waitUntil: 'networkidle0' });

  // let webfonts settle
  await page.evaluate(async () => {
    if (document.fonts && document.fonts.ready) await document.fonts.ready;
  });

  const stage = await page.$('#stage');

  console.log(`Capturing ${TOTAL_FRAMES} frames at ${FPS}fps (${DURATION}s loop)...`);
  for (let i = 0; i < TOTAL_FRAMES; i++) {
    const t = i / FPS;
    await page.evaluate((t) => window.renderAtTime(t), t);
    const framePath = path.join(FRAMES_DIR, `frame_${String(i).padStart(4, '0')}.png`);
    await stage.screenshot({ path: framePath });
    if (i % 30 === 0) console.log(`  frame ${i}/${TOTAL_FRAMES}`);
  }

  await browser.close();
  console.log('Capture complete. Encoding...');

  const framePattern = path.join(FRAMES_DIR, 'frame_%04d.png');

  // Animated WebP — full 24-bit color (no GIF-style 256-color palette/dithering).
  // True lossless was tested and rejected: at 1400x350/30fps/6s it lands at
  // 30-50MB (vs ~1.5MB lossy), way past a README's reasonable weight budget.
  // Instead we use libwebp's "text" preset, which is tuned for exactly this
  // kind of high-contrast type-on-gradient content and holds edges much
  // better than the default preset at the same quality level.
  execFileSync(ffmpegPath, [
    '-y',
    '-framerate', String(FPS),
    '-i', framePattern,
    '-loop', '0',
    '-vcodec', 'libwebp',
    '-lossless', '0',
    '-preset', '5', // "text" preset — sharper edges for type-heavy content
    '-q:v', '92',
    '-compression_level', '6',
    '-an',
    '-vsync', '0',
    OUT_WEBP,
  ], { stdio: 'inherit' });

  if (!process.env.KEEP_FRAMES) fs.rmSync(FRAMES_DIR, { recursive: true, force: true });

  console.log('Done.');
  console.log('  WebP:', OUT_WEBP, `(${(fs.statSync(OUT_WEBP).size / 1024).toFixed(0)} KB)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
