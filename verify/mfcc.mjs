// Standalone copy of the librosa-faithful MFCC from app/page.tsx for validation.
const SAMPLE_RATE = 2000, N_MFCC = 20, HOP_LENGTH = 128, N_FFT = 2048, N_MELS = 128, TOP_DB = 80;

function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k], aIm = im[i + k];
        const bRe = re[i + k + len / 2], bIm = im[i + k + len / 2];
        const tRe = bRe * curRe - bIm * curIm, tIm = bRe * curIm + bIm * curRe;
        re[i + k] = aRe + tRe; im[i + k] = aIm + tIm;
        re[i + k + len / 2] = aRe - tRe; im[i + k + len / 2] = aIm - tIm;
        const nRe = curRe * wRe - curIm * wIm; curIm = curRe * wIm + curIm * wRe; curRe = nRe;
      }
    }
  }
}
function hannPeriodic(N) { const w = new Float64Array(N); for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N)); return w; }
function hzToMelSlaney(f) { const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27; return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp; }
function melToHzSlaney(m) { const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp, logstep = Math.log(6.4) / 27; return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m; }
function buildMelFilterbank(sr, nFft, nMels) {
  const nBins = nFft / 2 + 1, fftFreqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * sr) / nFft;
  const minMel = hzToMelSlaney(0), maxMel = hzToMelSlaney(sr / 2), melF = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) melF[i] = melToHzSlaney(minMel + ((maxMel - minMel) * i) / (nMels + 1));
  const fdiff = new Float64Array(nMels + 1);
  for (let i = 0; i < nMels + 1; i++) fdiff[i] = melF[i + 1] - melF[i];
  const filters = [];
  for (let m = 0; m < nMels; m++) {
    const fb = new Float64Array(nBins), enorm = 2 / (melF[m + 2] - melF[m]);
    for (let k = 0; k < nBins; k++) {
      const lower = (fftFreqs[k] - melF[m]) / fdiff[m], upper = (melF[m + 2] - fftFreqs[k]) / fdiff[m + 1];
      fb[k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
    filters.push(fb);
  }
  return filters;
}
function buildDctMatrix(nMfcc, nMels) {
  const matrix = [], s0 = Math.sqrt(1 / (4 * nMels)), sk = Math.sqrt(1 / (2 * nMels));
  for (let k = 0; k < nMfcc; k++) {
    const row = new Float64Array(nMels), scale = k === 0 ? s0 : sk;
    for (let n = 0; n < nMels; n++) row[n] = 2 * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * nMels)) * scale;
    matrix.push(row);
  }
  return matrix;
}
const window = hannPeriodic(N_FFT), filters = buildMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS), dct = buildDctMatrix(N_MFCC, N_MELS);

export function extractMfcc(audio) {
  const nBins = N_FFT / 2 + 1, pad = Math.floor(N_FFT / 2);
  const padded = new Float64Array(audio.length + 2 * pad);
  for (let i = 0; i < audio.length; i++) padded[pad + i] = audio[i];
  const nFrames = 1 + Math.floor(audio.length / HOP_LENGTH);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  const melSpec = [];
  let globalMaxDb = -Infinity;
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) { re[i] = (padded[start + i] || 0) * window[i]; im[i] = 0; }
    fftRadix2(re, im);
    const mel = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      const fb = filters[m]; let sum = 0;
      for (let k = 0; k < nBins; k++) sum += (re[k] * re[k] + im[k] * im[k]) * fb[k];
      mel[m] = sum;
    }
    melSpec.push(mel);
  }
  const amin = 1e-10;
  for (let t = 0; t < nFrames; t++) { const mel = melSpec[t]; for (let m = 0; m < N_MELS; m++) { const db = 10 * Math.log10(Math.max(mel[m], amin)); mel[m] = db; if (db > globalMaxDb) globalMaxDb = db; } }
  const floorDb = globalMaxDb - TOP_DB;
  for (let t = 0; t < nFrames; t++) { const mel = melSpec[t]; for (let m = 0; m < N_MELS; m++) if (mel[m] < floorDb) mel[m] = floorDb; }
  const frames = [];
  for (let t = 0; t < nFrames; t++) {
    const mel = melSpec[t], out = new Float64Array(N_MFCC);
    for (let k = 0; k < N_MFCC; k++) { const row = dct[k]; let sum = 0; for (let n = 0; n < N_MELS; n++) sum += row[n] * mel[n]; out[k] = sum; }
    frames.push(out);
  }
  return frames;
}
