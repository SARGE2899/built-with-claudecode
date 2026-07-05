// Deterministic, time-driven animation. Every visual property is a pure
// function of `t` (seconds) so the loop can be captured frame-by-frame
// with Puppeteer and always looks identical to real-time playback.

const DURATION = 6.0; // seconds — full loop length

const easeOutExpo = gsap.parseEase('expo.out');
const easeInPow2 = gsap.parseEase('power2.in');
const easeInOutPow2 = gsap.parseEase('power2.inOut');
const easeOutBack = gsap.parseEase('back.out(1.4)');

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const progress = (t, start, dur) => clamp01((t - start) / dur);

// ---- shared exit envelope: all headline words fade out together ----
const EXIT_START = 4.5;
const EXIT_DUR = 0.8;

function wordState(t, enterStart, enterDur, useBack) {
  const enterP = progress(t, enterStart, enterDur);
  const enterE = (useBack ? easeOutBack : easeOutExpo)(enterP);

  const exitP = progress(t, EXIT_START, EXIT_DUR);
  const exitE = easeInPow2(exitP);

  const opacity = enterE * (1 - exitE);
  const translateY = (1 - enterE) * 18 + exitE * -12;
  const blur = (1 - enterE) * 10 + exitE * 7;
  const scale = 1 - (1 - enterE) * 0.04;

  return { opacity, translateY, blur, scale };
}

function applyWord(el, state) {
  el.style.opacity = state.opacity;
  el.style.transform = `translateY(${state.translateY}px) scale(${state.scale})`;
  el.style.filter = `blur(${state.blur}px)`;
}

// ---- background blobs: pure periodic motion, period divides DURATION ----
function blobTransform(t, period, radiusX, radiusY, phase) {
  const angle = (2 * Math.PI * t) / period + phase;
  const x = Math.cos(angle) * radiusX;
  const y = Math.sin(angle) * radiusY;
  return `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
}

function breathe(t, period, min, max, phase = 0) {
  const s = (Math.sin((2 * Math.PI * t) / period + phase) + 1) / 2;
  return min + (max - min) * s;
}

// ---- underline draw / retract ----
const UNDERLINE_LEN = 412;
const DRAW_START = 1.55;
const DRAW_DUR = 0.62;
const RETRACT_START = 4.3;
const RETRACT_DUR = 0.55;

function underlineState(t) {
  const drawP = progress(t, DRAW_START, DRAW_DUR);
  const drawE = easeInOutPow2(drawP);
  const retractP = progress(t, RETRACT_START, RETRACT_DUR);
  const retractE = easeInOutPow2(retractP);
  const drawnFraction = drawE * (1 - retractE);
  return drawnFraction;
}

// ---- shimmer sweep across "Claude Code" ----
const SHIMMER_START = 2.55;
const SHIMMER_DUR = 1.0;

function shimmerState(t) {
  const p = progress(t, SHIMMER_START, SHIMMER_DUR);
  const e = easeInOutPow2(p);
  const xPercent = -70 + e * 190;
  const envelope = Math.sin(Math.max(0, Math.min(1, p)) * Math.PI);
  return { xPercent, opacity: envelope * 0.9 };
}

// ---- DOM refs ----
const wBuilt = document.getElementById('wBuilt');
const wWith = document.getElementById('wWith');
const wClaude = document.getElementById('wClaude');
const shimmerEl = document.getElementById('shimmer');
const blobA = document.getElementById('blobA');
const blobB = document.getElementById('blobB');
const blobC = document.getElementById('blobC');
const underlineDraw = document.getElementById('underlineDraw');
const underlineDot = document.getElementById('underlineDot');
const underlineTrack = document.getElementById('underlineTrack');
const underlinePath = underlineDraw;

// content envelope — same entrance/exit shape as the headline, used to
// keep the underline track from lingering alone during the quiet tail
function contentEnvelope(t) {
  const enterP = progress(t, 0.30, 0.9);
  const enterE = easeOutExpo(enterP);
  const exitP = progress(t, EXIT_START, EXIT_DUR);
  const exitE = easeInPow2(exitP);
  return enterE * (1 - exitE);
}

function renderAtTime(t) {
  t = t % DURATION;

  // headline words — staggered entrance, unified exit
  applyWord(wBuilt, wordState(t, 0.30, 0.68, true));
  applyWord(wWith, wordState(t, 0.52, 0.68, true));
  applyWord(wClaude, wordState(t, 0.78, 0.78, true));

  // background blobs — slow independent orbits, all periods divide DURATION
  blobA.style.transform = blobTransform(t, 6.0, 46, 34, 0);
  blobB.style.transform = blobTransform(t, 6.0, 38, 42, Math.PI * 0.66);
  blobC.style.transform = blobTransform(t, 3.0, 26, 20, Math.PI * 1.2);

  blobA.style.opacity = breathe(t, 3.0, 0.55, 0.85, 0);
  blobB.style.opacity = breathe(t, 3.0, 0.45, 0.7, Math.PI * 0.5);
  blobC.style.opacity = breathe(t, 2.0, 0.35, 0.55, Math.PI);

  // underline draw + traveling dot
  underlineTrack.style.opacity = (contentEnvelope(t) * 0.18).toFixed(3);

  const frac = underlineState(t);
  const offset = UNDERLINE_LEN * (1 - frac);
  underlineDraw.style.strokeDashoffset = offset.toFixed(2);

  if (frac > 0.002 && frac < 0.998) {
    const pt = underlinePath.getPointAtLength(frac * UNDERLINE_LEN);
    underlineDot.setAttribute('cx', pt.x.toFixed(2));
    underlineDot.setAttribute('cy', pt.y.toFixed(2));
    underlineDot.setAttribute('opacity', '1');
  } else {
    underlineDot.setAttribute('opacity', '0');
  }

  // shimmer sweep
  const sh = shimmerState(t);
  shimmerEl.style.transform = `translateX(${sh.xPercent}%) skewX(-12deg)`;
  shimmerEl.style.opacity = sh.opacity;
}

window.renderAtTime = renderAtTime;
window.DURATION = DURATION;

// live preview when opened directly in a browser
if (!window.__CAPTURE_MODE__) {
  let start = null;
  function tick(ts) {
    if (start === null) start = ts;
    const t = ((ts - start) / 1000) % DURATION;
    renderAtTime(t);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
