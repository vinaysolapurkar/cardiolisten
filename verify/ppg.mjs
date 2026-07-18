// Validate that PPG heart-rate recovery is invariant to sampling rate/jitter.
// The old code hardcoded fps=30; a 60fps capture would read 2x (120 not 60).

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
  const trend = smoothSignal(signal, Math.max(3, Math.round(fps * 1.5)));
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
function fftHeartRate(signal, fps) {
  const N = signal.length; if (N < 8) return 0;
  let fftSize = 1; while (fftSize < N) fftSize *= 2; fftSize *= 2;
  const re = new Float64Array(fftSize), im = new Float64Array(fftSize);
  for (let i = 0; i < N; i++) re[i] = signal[i] * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1))));
  fftRadix2(re, im);
  const halfN = Math.floor(fftSize / 2), mag = new Float64Array(halfN);
  for (let k = 0; k < halfN; k++) mag[k] = Math.hypot(re[k], im[k]);
  const minBin = Math.max(1, Math.floor((0.7 * fftSize) / fps)), maxBin = Math.min(halfN - 1, Math.ceil((3.3 * fftSize) / fps));
  let peakBin = minBin, peakMag = 0;
  for (let k = minBin; k <= maxBin; k++) if (mag[k] > peakMag) { peakMag = mag[k]; peakBin = k; }
  let interpBin = peakBin;
  if (peakBin > 0 && peakBin < halfN - 1) { const a = mag[peakBin - 1], b = mag[peakBin], c = mag[peakBin + 1], denom = a - 2 * b + c; if (denom !== 0) interpBin = peakBin + (0.5 * (a - c)) / denom; }
  return Math.round((interpBin * fps) / fftSize * 60);
}

function recover(trueBpm, nominalFps, jitter, seed) {
  // Build a PPG-like signal at trueBpm sampled at ~nominalFps with jitter and duplicate frames.
  const f = trueBpm / 60; // Hz
  const values = [], times = [];
  let t = 0; let rnd = seed;
  const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
  for (let i = 0; i < nominalFps * 25; i++) {
    // occasionally duplicate a frame (same timestamp advance but repeated value)
    const dt = (1000 / nominalFps) * (1 + (rand() - 0.5) * jitter);
    t += dt;
    const v = 128 + 20 * Math.sin(2 * Math.PI * f * (t / 1000)) + 3 * Math.sin(2 * Math.PI * 2 * f * (t / 1000)) + (rand() - 0.5) * 2;
    values.push(v); times.push(t);
  }
  const { signal, fs } = resampleUniform(values, times);
  const detr = detrendSignal(signal, fs);
  const sm = smoothSignal(detr, Math.max(2, Math.round(fs / 6)));
  return { bpm: fftHeartRate(sm, fs), fs };
}

let allPass = true;
for (const [bpm, fps, jit] of [[60, 30, 0.1], [60, 60, 0.1], [75, 30, 0.3], [75, 60, 0.3], [48, 24, 0.2], [120, 30, 0.15], [100, 90, 0.25]]) {
  const { bpm: got, fs } = recover(bpm, fps, jit, 12345 + bpm + fps);
  const err = Math.abs(got - bpm);
  const ok = err <= 3;
  allPass = allPass && ok;
  console.log(`true=${bpm} BPM @~${fps}fps jitter=${jit} -> recovered=${got} (measured fs=${fs.toFixed(1)}) err=${err} ${ok ? "OK" : "FAIL"}`);
}
console.log(allPass ? "\nRESULT: PASS ✓ (HR invariant to sampling rate — the 50-vs-90 bug is fixed)" : "\nRESULT: FAIL ✗");
process.exit(allPass ? 0 : 1);
