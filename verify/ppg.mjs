// Validate PPG heart-rate recovery: (a) invariant to sampling rate/jitter, and
// (b) robust to large low-frequency motion artifacts (post-exercise shaky hand),
// which is what made a running user read 46 BPM.

function smoothSignal(signal, windowSize) {
  const result = new Array(signal.length).fill(0);
  const half = Math.floor(windowSize / 2);
  for (let i = 0; i < signal.length; i++) {
    const start = Math.max(0, i - half), end = Math.min(signal.length, i + half + 1);
    let sum = 0; for (let j = start; j < end; j++) sum += signal[j];
    result[i] = sum / (end - start);
  }
  return result;
}
function detrendSignal(signal, fps) {
  // stronger high-pass (1.0s) to suppress slow motion/breathing sway
  const trend = smoothSignal(signal, Math.max(3, Math.round(fps * 1.0)));
  return signal.map((v, i) => v - trend[i]);
}
function resampleUniform(values, timesMs) {
  const n = values.length;
  if (n < 4) return { signal: values.slice(), fs: 30 };
  const durationS = (timesMs[n - 1] - timesMs[0]) / 1000;
  const dts = []; for (let i = 1; i < n; i++) { const dt = timesMs[i] - timesMs[i - 1]; if (dt > 0) dts.push(dt); }
  dts.sort((a, b) => a - b);
  const medDt = dts[Math.floor(dts.length / 2)] || 33.3;
  const fs = Math.min(120, Math.max(15, 1000 / medDt));
  const m = Math.max(4, Math.floor(durationS * fs));
  const out = new Array(m); let j = 0;
  for (let i = 0; i < m; i++) {
    const t = timesMs[0] + (i / fs) * 1000;
    while (j < n - 2 && timesMs[j + 1] < t) j++;
    const t0 = timesMs[j], t1 = timesMs[j + 1], v0 = values[j], v1 = values[j + 1];
    const frac = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    out[i] = v0 + (v1 - v0) * Math.max(0, Math.min(1, frac));
  }
  return { signal: out, fs };
}
function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k], aIm = im[i + k], bRe = re[i + k + len / 2], bIm = im[i + k + len / 2];
        const tRe = bRe * curRe - bIm * curIm, tIm = bRe * curIm + bIm * curRe;
        re[i + k] = aRe + tRe; im[i + k] = aIm + tIm; re[i + k + len / 2] = aRe - tRe; im[i + k + len / 2] = aIm - tIm;
        const nRe = curRe * wRe - curIm * wIm; curIm = curRe * wIm + curIm * wRe; curRe = nRe;
      } }
  }
}
function spectrum(signal, fps) {
  const N = signal.length;
  let fftSize = 1; while (fftSize < N) fftSize *= 2; fftSize *= 2;
  const re = new Float64Array(fftSize), im = new Float64Array(fftSize);
  for (let i = 0; i < N; i++) re[i] = signal[i] * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1))));
  fftRadix2(re, im);
  const halfN = Math.floor(fftSize / 2), mag = new Float64Array(halfN);
  for (let k = 0; k < halfN; k++) mag[k] = Math.hypot(re[k], im[k]);
  return { mag, fftSize, halfN };
}
// OLD: raw global peak in band
function hrOld(signal, fps) {
  if (signal.length < 8) return 0;
  const { mag, fftSize, halfN } = spectrum(signal, fps);
  const minBin = Math.max(1, Math.floor((0.7 * fftSize) / fps)), maxBin = Math.min(halfN - 1, Math.ceil((3.3 * fftSize) / fps));
  let peakBin = minBin, peakMag = 0;
  for (let k = minBin; k <= maxBin; k++) if (mag[k] > peakMag) { peakMag = mag[k]; peakBin = k; }
  return Math.round((peakBin * fps) / fftSize * 60);
}
// NEW: harmonic-product. score(k) = mag[k] * mag[2k]. A real pulse is sharply
// non-sinusoidal so it always has a 2nd harmonic; a lone motion/breathing peak
// does not, so it scores ~0. This rejects both low-frequency motion artifacts
// and octave-halving. (Deliberately NOT using the 3rd harmonic: motion at ~f/3
// would put its 3rd-harmonic bin on top of the true pulse and steal credit.)
function hrHarmonic(signal, fps) {
  if (signal.length < 8) return { bpm: 0, quality: 0 };
  const { mag, fftSize, halfN } = spectrum(signal, fps);
  const minBin = Math.max(1, Math.floor((0.7 * fftSize) / fps)), maxBin = Math.min(halfN - 1, Math.ceil((3.3 * fftSize) / fps));
  let bestBin = minBin, bestScore = -1, sumMag = 0, cnt = 0, peakMag = 0;
  for (let k = minBin; k <= maxBin; k++) { sumMag += mag[k]; cnt++; if (mag[k] > peakMag) peakMag = mag[k]; }
  for (let k = minBin; k <= maxBin; k++) {
    const h2 = 2 * k < halfN ? mag[2 * k] : 0;
    const score = mag[k] * h2;
    if (score > bestScore) { bestScore = score; bestBin = k; }
  }
  // parabolic interpolation on raw magnitude around the chosen fundamental
  let interp = bestBin;
  if (bestBin > 0 && bestBin < halfN - 1) {
    const a = mag[bestBin - 1], b = mag[bestBin], c = mag[bestBin + 1], denom = a - 2 * b + c;
    if (denom !== 0) interp = bestBin + (0.5 * (a - c)) / denom;
  }
  const meanMag = cnt > 0 ? sumMag / cnt : 1;
  const quality = meanMag > 0 ? Math.min(1, (mag[bestBin] / meanMag - 1) / 8) : 0;
  return { bpm: Math.round((interp * fps) / fftSize * 60), quality: Math.max(0, quality) };
}

function build(trueBpm, fps, jitter, motionHz, motionAmp, seed) {
  const f = trueBpm / 60;
  const values = [], times = [];
  let t = 0, rnd = seed;
  const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
  for (let i = 0; i < fps * 25; i++) {
    t += (1000 / fps) * (1 + (rand() - 0.5) * jitter);
    const ts = t / 1000;
    const pulse = 20 * Math.sin(2 * Math.PI * f * ts) + 6 * Math.sin(2 * Math.PI * 2 * f * ts) + 2 * Math.sin(2 * Math.PI * 3 * f * ts);
    const motion = motionAmp ? motionAmp * Math.sin(2 * Math.PI * motionHz * ts) : 0;
    values.push(128 + pulse + motion + (rand() - 0.5) * 2);
    times.push(t);
  }
  const { signal, fs } = resampleUniform(values, times);
  const detr = detrendSignal(signal, fs);
  const sm = smoothSignal(detr, Math.max(2, Math.round(fs / 8)));
  return { sm, fs };
}

// [bpm, fps, jitter, motionHz, motionAmp, label]
const CASES = [
  [60, 30, 0.1, 0, 0, "resting 60 @30fps"],
  [60, 60, 0.1, 0, 0, "resting 60 @60fps"],
  [75, 30, 0.3, 0, 0, "75 @30fps jitter"],
  [48, 24, 0.2, 0, 0, "low 48 @24fps"],
  [120, 30, 0.15, 0, 0, "high 120 @30fps"],
  [140, 30, 0.2, 0.8, 60, "POST-EXERCISE 140 + big 0.8Hz motion"],
  [150, 30, 0.2, 1.0, 50, "POST-EXERCISE 150 + big 1.0Hz motion"],
  [95, 30, 0.2, 0.7, 45, "walking 95 + 0.7Hz sway"],
];

let allPass = true;
for (const [bpm, fps, jit, mHz, mAmp, label] of CASES) {
  const { sm, fs } = build(bpm, fps, jit, mHz, mAmp, 1234 + bpm + fps);
  const old = hrOld(sm, fs);
  const nw = hrHarmonic(sm, fs);
  const ok = Math.abs(nw.bpm - bpm) <= 4;
  allPass = allPass && ok;
  console.log(`${label.padEnd(42)} old=${String(old).padStart(3)} new=${String(nw.bpm).padStart(3)} (q=${nw.quality.toFixed(2)}) want=${bpm} ${ok ? "OK" : "FAIL"}`);
}
console.log(allPass ? "\nRESULT: PASS ✓ (harmonic-aware HR is accurate incl. post-exercise motion)" : "\nRESULT: FAIL ✗");
process.exit(allPass ? 0 : 1);
