"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// ============ CONSTANTS ============
// Audio / model (MUST match train_model.py + librosa defaults)
const SAMPLE_RATE = 2000;      // model was trained on librosa.load(sr=2000)
const SEGMENT_DURATION = 3;    // seconds
const RECORD_DURATION = 30;    // seconds of chest recording
const N_MFCC = 20;
const HOP_LENGTH = 128;
const N_FFT = 2048;            // librosa.feature.mfcc default n_fft
const N_MELS = 128;            // librosa.feature.melspectrogram default n_mels
const MAX_FRAMES = 47;         // 1 + floor(SAMPLE_RATE*SEGMENT_DURATION / HOP_LENGTH)
const TOP_DB = 80;             // librosa.power_to_db default

// PPG (pulse)
const PPG_DURATION = 30;       // seconds of finger reading
const WAVEFORM_POINTS = 300;

type Tab = "pulse" | "sound" | "report";
type PulseState = "idle" | "measuring" | "processing" | "done";
type SoundState = "idle" | "recording" | "processing" | "done";

interface PulseResult {
  hr: number;
  hrv: number;
  rrIntervals: number[];
  irregularity: number;
  rhythm: "regular" | "irregular" | "highly_irregular";
  quality: number; // 0..1 signal quality
  fs: number;      // measured sampling rate
  signal: number[];
}

interface SoundResult {
  label: string;
  confidence: number;
  normal: number;
  abnormal: number;
  segmentsAnalyzed: number;
  quality: number; // 0..1 best-window quality
}

// On-device diagnostics so real hardware failures can be diagnosed from a
// screenshot instead of guessed at. Not used in the HR/label math.
interface PulseDiag {
  torchCapable: boolean;
  exposureCapable: boolean;
  exposureLockState: "n/a" | "locked" | "failed" | "pending";
  r: number; g: number; b: number;
  fps: number;
  ampPP: number; // peak-to-peak of the filtered signal, last ~2s
}
interface SoundDiag {
  level: number; // 0..1 live RMS of raw mic input
  peak: number;  // 0..1 running max
}

// ============ FFT (iterative radix-2, in-place) ============
// Operates on interleaved-free re/im arrays; length must be a power of 2.
function fftRadix2(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k], aIm = im[i + k];
        const bRe = re[i + k + len / 2], bIm = im[i + k + len / 2];
        const tRe = bRe * curRe - bIm * curIm;
        const tIm = bRe * curIm + bIm * curRe;
        re[i + k] = aRe + tRe;
        im[i + k] = aIm + tIm;
        re[i + k + len / 2] = aRe - tRe;
        im[i + k + len / 2] = aIm - tIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
}

// ============ librosa-faithful MFCC ============

// Periodic Hann window (scipy get_window('hann', N, fftbins=True))
function hannPeriodic(N: number): Float64Array {
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
  return w;
}

// Slaney mel scale (librosa default htk=False)
function hzToMelSlaney(f: number): number {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp; // 15.0
  const logstep = Math.log(6.4) / 27.0;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fSp;
}
function melToHzSlaney(m: number): number {
  const fSp = 200.0 / 3;
  const minLogHz = 1000.0;
  const minLogMel = minLogHz / fSp;
  const logstep = Math.log(6.4) / 27.0;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fSp * m;
}

// librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm='slaney')
function buildMelFilterbank(sr: number, nFft: number, nMels: number): Float64Array[] {
  const nBins = nFft / 2 + 1;
  const fftFreqs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) fftFreqs[i] = (i * sr) / nFft;

  const fmin = 0, fmax = sr / 2;
  const minMel = hzToMelSlaney(fmin);
  const maxMel = hzToMelSlaney(fmax);
  const melF = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    const mel = minMel + ((maxMel - minMel) * i) / (nMels + 1);
    melF[i] = melToHzSlaney(mel);
  }
  const fdiff = new Float64Array(nMels + 1);
  for (let i = 0; i < nMels + 1; i++) fdiff[i] = melF[i + 1] - melF[i];

  const filters: Float64Array[] = [];
  for (let m = 0; m < nMels; m++) {
    const fb = new Float64Array(nBins);
    const enorm = 2.0 / (melF[m + 2] - melF[m]); // Slaney normalization
    for (let k = 0; k < nBins; k++) {
      const lower = (fftFreqs[k] - melF[m]) / fdiff[m];
      const upper = (melF[m + 2] - fftFreqs[k]) / fdiff[m + 1];
      const val = Math.max(0, Math.min(lower, upper));
      fb[k] = val * enorm;
    }
    filters.push(fb);
  }
  return filters;
}

// Ortho DCT-II matrix (scipy dct type=2 norm='ortho'), first nMfcc rows
function buildDctMatrix(nMfcc: number, nMels: number): Float64Array[] {
  const matrix: Float64Array[] = [];
  const s0 = Math.sqrt(1 / (4 * nMels));
  const sk = Math.sqrt(1 / (2 * nMels));
  for (let k = 0; k < nMfcc; k++) {
    const row = new Float64Array(nMels);
    const scale = k === 0 ? s0 : sk;
    for (let n = 0; n < nMels; n++) {
      row[n] = 2 * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * nMels)) * scale;
    }
    matrix.push(row);
  }
  return matrix;
}

// Precompute reusable filterbanks/matrices/window once.
let _mfccCache: {
  window: Float64Array;
  filters: Float64Array[];
  dct: Float64Array[];
} | null = null;
function mfccResources() {
  if (!_mfccCache) {
    _mfccCache = {
      window: hannPeriodic(N_FFT),
      filters: buildMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS),
      dct: buildDctMatrix(N_MFCC, N_MELS),
    };
  }
  return _mfccCache;
}

// Faithful librosa.feature.mfcc(y, sr=2000, n_mfcc=20, hop=128) -> [frames][nMfcc]
function extractMfcc(audio: Float32Array): Float64Array[] {
  const { window, filters, dct } = mfccResources();
  const nBins = N_FFT / 2 + 1;
  const pad = Math.floor(N_FFT / 2);

  // Center padding (pad_mode='constant'/zero — the librosa >=0.10 STFT default)
  const padded = new Float64Array(audio.length + 2 * pad);
  for (let i = 0; i < audio.length; i++) padded[pad + i] = audio[i];

  const nFrames = 1 + Math.floor(audio.length / HOP_LENGTH);
  const re = new Float64Array(N_FFT);
  const im = new Float64Array(N_FFT);

  // First compute mel power spectrogram for whole segment (needed for top_db max)
  const melSpec: Float64Array[] = [];
  let globalMaxDb = -Infinity;
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP_LENGTH;
    for (let i = 0; i < N_FFT; i++) {
      re[i] = (padded[start + i] ?? 0) * window[i];
      im[i] = 0;
    }
    fftRadix2(re, im);
    // power spectrum (|X|^2), power=2
    const mel = new Float64Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      const fb = filters[m];
      let sum = 0;
      for (let k = 0; k < nBins; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        sum += p * fb[k];
      }
      mel[m] = sum;
    }
    melSpec.push(mel);
  }

  // power_to_db: 10*log10(max(S, amin)); ref=1
  const amin = 1e-10;
  for (let t = 0; t < nFrames; t++) {
    const mel = melSpec[t];
    for (let m = 0; m < N_MELS; m++) {
      const db = 10 * Math.log10(Math.max(mel[m], amin));
      mel[m] = db;
      if (db > globalMaxDb) globalMaxDb = db;
    }
  }
  // top_db clamp against the global max
  const floorDb = globalMaxDb - TOP_DB;
  for (let t = 0; t < nFrames; t++) {
    const mel = melSpec[t];
    for (let m = 0; m < N_MELS; m++) if (mel[m] < floorDb) mel[m] = floorDb;
  }

  // DCT-II ortho, keep first N_MFCC
  const frames: Float64Array[] = [];
  for (let t = 0; t < nFrames; t++) {
    const mel = melSpec[t];
    const out = new Float64Array(N_MFCC);
    for (let k = 0; k < N_MFCC; k++) {
      const row = dct[k];
      let sum = 0;
      for (let n = 0; n < N_MELS; n++) sum += row[n] * mel[n];
      out[k] = sum;
    }
    frames.push(out);
  }
  return frames;
}

// ============ Audio resampling (anti-aliased) ============
// Block-average decimation acts as a crude anti-alias lowpass (far better than
// the previous nearest-sample decimation which aliased heavily).
function resampleAudio(input: Float32Array, srcRate: number, dstRate: number): Float32Array {
  if (srcRate === dstRate) return input;
  const ratio = srcRate / dstRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0, cnt = 0;
    for (let j = start; j < end; j++) { sum += input[j]; cnt++; }
    out[i] = cnt > 0 ? sum / cnt : 0;
  }
  // remove DC offset
  let mean = 0;
  for (let i = 0; i < out.length; i++) mean += out[i];
  mean /= out.length || 1;
  for (let i = 0; i < out.length; i++) out[i] -= mean;
  return out;
}

// ============ Heart sound window quality scoring ============
// Shared by: (a) the per-zone 10s pass/fail gate, and (b) the final best-5
// window selection before model inference. Moderate, non-clipping RMS
// combined with low coefficient-of-variation of energy across 10 sub-chunks
// (steady recording condition, not jumpy handling noise) scores highest.

interface WindowScore { score: number; rms: number }
interface ScoredWindow extends WindowScore { start: number }

function scoreAudioWindow(segment: Float32Array | Float64Array): WindowScore {
  let rms = 0;
  for (let i = 0; i < segment.length; i++) rms += segment[i] * segment[i];
  rms = Math.sqrt(rms / segment.length);
  if (rms < 0.0005) return { score: 0, rms };

  const chunkSize = Math.floor(segment.length / 10);
  const chunkEnergies: number[] = [];
  for (let c = 0; c < 10; c++) {
    let e = 0;
    for (let i = c * chunkSize; i < (c + 1) * chunkSize; i++) e += segment[i] * segment[i];
    chunkEnergies.push(e / chunkSize);
  }
  const meanEnergy = chunkEnergies.reduce((a, b) => a + b, 0) / chunkEnergies.length;
  const variance = chunkEnergies.reduce((s, e) => s + (e - meanEnergy) ** 2, 0) / chunkEnergies.length;
  const cv = Math.sqrt(variance) / (meanEnergy + 1e-10);

  const rmsScore = rms > 0.001 && rms < 0.5 ? 1 : 0.3;
  const cvScore = Math.max(0, 1 - cv * 2);
  return { score: rmsScore * cvScore, rms };
}

// Slides scoreAudioWindow across `clip` at the given window/hop size
// (seconds). Skips near-silent windows (rms < 0.0005) entirely — matches the
// original inline behavior. Returns every kept window plus how many total
// slide positions were attempted (kept + skipped), for debug reporting.
function scoreAudioWindows(clip: Float32Array, sampleRate: number, windowSeconds: number, hopSeconds: number): { windows: ScoredWindow[]; totalWindows: number } {
  const windowSize = Math.round(sampleRate * windowSeconds);
  const hopSize = Math.max(1, Math.round(sampleRate * hopSeconds));
  const windows: ScoredWindow[] = [];
  let totalWindows = 0;
  for (let start = 0; start + windowSize <= clip.length; start += hopSize) {
    totalWindows++;
    const s = scoreAudioWindow(clip.subarray(start, start + windowSize));
    if (s.rms < 0.0005) continue;
    windows.push({ start, ...s });
  }
  return { windows, totalWindows };
}

// Best (highest-score) window found in `clip`; { score: 0, rms: 0 } if none.
function bestWindowScore(clip: Float32Array, sampleRate: number, windowSeconds: number, hopSeconds: number): WindowScore {
  const { windows } = scoreAudioWindows(clip, sampleRate, windowSeconds, hopSeconds);
  let best: WindowScore = { score: 0, rms: 0 };
  for (const w of windows) if (w.score > best.score) best = w;
  return best;
}

// ============ Live beat envelope detection ============
// Purely visual — drives the ECG-style spike trace only. Does NOT feed the
// pass/fail gate or the model input, so it doesn't need to be clinically
// precise, only responsive and honest (no fake/randomized spikes).
//
// Tuned for ~60 calls/second (one call per animation frame), NOT per raw
// audio sample — the caller (Task 4) feeds one representative amplitude
// scalar per frame, not the full audio buffer.

interface BeatEnvelopeState {
  fastEnv: number;   // fast-attack/decay envelope of the rectified signal
  slowEnv: number;   // slow rolling average; sets the adaptive threshold
  lastBeatMs: number; // timestamp (ms) of the last fired beat
}

function initBeatEnvelopeState(): BeatEnvelopeState {
  return { fastEnv: 0, slowEnv: 0, lastBeatMs: -Infinity };
}

// Feed one rectified (Math.abs) amplitude sample at time `nowMs`. A beat
// fires when the fast envelope exceeds 1.8x the slow envelope, with a 150ms
// refractory period — short enough to catch S1 ("lub") and S2 ("dub") as two
// separate spikes per cardiac cycle, matching a real heart-sound trace.
function processBeatEnvelopeSample(state: BeatEnvelopeState, sampleAbs: number, nowMs: number): { state: BeatEnvelopeState; beatDetected: boolean } {
  const fastAlpha = 0.5;
  const slowAlpha = 0.015;
  const fastEnv = state.fastEnv + fastAlpha * (sampleAbs - state.fastEnv);
  const slowEnv = state.slowEnv + slowAlpha * (sampleAbs - state.slowEnv);

  const threshold = Math.max(slowEnv * 1.8, 0.01);
  const refractoryOk = nowMs - state.lastBeatMs > 150;
  const beatDetected = fastEnv > threshold && refractoryOk;

  return {
    state: { fastEnv, slowEnv, lastBeatMs: beatDetected ? nowMs : state.lastBeatMs },
    beatDetected,
  };
}

// ============ PPG Signal Processing ============

function smoothSignal(signal: number[], windowSize: number): number[] {
  const result = new Array(signal.length).fill(0);
  const half = Math.floor(windowSize / 2);
  for (let i = 0; i < signal.length; i++) {
    const start = Math.max(0, i - half);
    const end = Math.min(signal.length, i + half + 1);
    let sum = 0;
    for (let j = start; j < end; j++) sum += signal[j];
    result[i] = sum / (end - start);
  }
  return result;
}

function detrendSignal(signal: number[], fps: number): number[] {
  // ~1s moving-average high-pass to suppress slow motion/breathing sway
  const trendWindow = Math.max(3, Math.round(fps * 1.0));
  const trend = smoothSignal(signal, trendWindow);
  return signal.map((v, i) => v - trend[i]);
}

// Resample irregular (value,timeMs) samples onto a uniform grid.
// Returns the uniform signal and the true sampling rate used.
function resampleUniform(values: number[], timesMs: number[]): { signal: number[]; fs: number } {
  const n = values.length;
  if (n < 4) return { signal: values.slice(), fs: 30 };
  const durationS = (timesMs[n - 1] - timesMs[0]) / 1000;
  if (durationS <= 0) return { signal: values.slice(), fs: 30 };

  const dts: number[] = [];
  for (let i = 1; i < n; i++) { const dt = timesMs[i] - timesMs[i - 1]; if (dt > 0) dts.push(dt); }
  dts.sort((a, b) => a - b);
  const medDt = dts[Math.floor(dts.length / 2)] || 33.3;
  const fs = Math.min(120, Math.max(15, 1000 / medDt));

  const m = Math.max(4, Math.floor(durationS * fs));
  const out = new Array(m);
  let j = 0;
  for (let i = 0; i < m; i++) {
    const t = timesMs[0] + (i / fs) * 1000;
    while (j < n - 2 && timesMs[j + 1] < t) j++;
    const t0 = timesMs[j], t1 = timesMs[j + 1], v0 = values[j], v1 = values[j + 1];
    const frac = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
    out[i] = v0 + (v1 - v0) * Math.max(0, Math.min(1, frac));
  }
  return { signal: out, fs };
}

// Heart rate from a uniformly-sampled signal at known fps, using a
// harmonic-product criterion: score(k) = mag[k] * mag[2k]. A real pulse is
// sharply non-sinusoidal so it always carries a 2nd harmonic; a lone
// low-frequency motion/breathing peak (shaky hand after exercise) does not, so
// it scores ~0 and is rejected. This also rejects octave-halving. We stop at
// the 2nd harmonic on purpose: including the 3rd would let motion near f/3 drop
// its harmonic onto the true pulse and steal the reading.
function fftHeartRate(signal: number[], fps: number): { bpm: number; quality: number } {
  const N = signal.length;
  if (N < 8) return { bpm: 0, quality: 0 };
  let fftSize = 1;
  while (fftSize < N) fftSize *= 2;
  fftSize *= 2; // extra zero-padding for finer bin resolution

  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  for (let i = 0; i < N; i++) re[i] = signal[i] * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1))));
  fftRadix2(re, im);

  const halfN = Math.floor(fftSize / 2);
  const mag = new Float64Array(halfN);
  for (let k = 0; k < halfN; k++) mag[k] = Math.hypot(re[k], im[k]);

  // Search fundamentals in 0.7-3.3 Hz (42-198 BPM)
  const minBin = Math.max(1, Math.floor((0.7 * fftSize) / fps));
  const maxBin = Math.min(halfN - 1, Math.ceil((3.3 * fftSize) / fps));
  let bestBin = minBin, bestScore = -1, sumMag = 0, cnt = 0;
  for (let k = minBin; k <= maxBin; k++) { sumMag += mag[k]; cnt++; }
  for (let k = minBin; k <= maxBin; k++) {
    const h2 = 2 * k < halfN ? mag[2 * k] : 0;
    const score = mag[k] * h2;
    if (score > bestScore) { bestScore = score; bestBin = k; }
  }
  const meanMag = cnt > 0 ? sumMag / cnt : 1;
  const quality = meanMag > 0 ? Math.min(1, (mag[bestBin] / meanMag - 1) / 8) : 0;

  // Parabolic interpolation on the raw magnitude around the chosen fundamental
  let interpBin = bestBin;
  if (bestBin > 0 && bestBin < halfN - 1) {
    const a = mag[bestBin - 1], b = mag[bestBin], c = mag[bestBin + 1];
    const denom = a - 2 * b + c;
    if (denom !== 0) interpBin = bestBin + (0.5 * (a - c)) / denom;
  }
  const freqHz = (interpBin * fps) / fftSize;
  return { bpm: Math.round(freqHz * 60), quality: Math.max(0, quality) };
}

function findPeaks(signal: number[], minDistance: number): number[] {
  const peaks: number[] = [];
  for (let i = 2; i < signal.length - 2; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] &&
        signal[i] > signal[i - 2] && signal[i] > signal[i + 2]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) peaks.push(i);
    }
  }
  return peaks;
}

function analyzePPG(values: number[], timesMs: number[]): PulseResult {
  // Resample onto a true uniform grid at the measured rate
  const { signal: uniform, fs } = resampleUniform(values, timesMs);

  // Discard first ~2s (auto-exposure / settling)
  const settle = Math.min(Math.round(fs * 2), Math.floor(uniform.length * 0.15));
  const trimmed = uniform.slice(settle);

  const detrended = detrendSignal(trimmed, fs);
  const smoothed = smoothSignal(detrended, Math.max(2, Math.round(fs / 8)));

  const { bpm: hr, quality } = fftHeartRate(smoothed, fs);

  // Peak detection for HRV/rhythm, gated by the FFT HR
  const minDist = Math.max(3, Math.round(fs * (60 / Math.min(hr * 1.4, 200))));
  const peaks = findPeaks(smoothed, minDist);
  const rrIntervals: number[] = [];
  const expectedRR = hr > 0 ? 60000 / hr : 1000;
  for (let i = 1; i < peaks.length; i++) {
    const rrMs = ((peaks[i] - peaks[i - 1]) / fs) * 1000;
    if (rrMs > expectedRR * 0.6 && rrMs < expectedRR * 1.4) rrIntervals.push(rrMs);
  }

  let sumSqDiff = 0, validDiffs = 0;
  for (let i = 1; i < rrIntervals.length; i++) {
    const diff = rrIntervals[i] - rrIntervals[i - 1];
    sumSqDiff += diff * diff; validDiffs++;
  }
  const hrv = validDiffs > 0 ? Math.round(Math.sqrt(sumSqDiff / validDiffs)) : 0;

  const meanRR = rrIntervals.length > 0 ? rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length : expectedRR;
  const stdRR = rrIntervals.length > 1 ? Math.sqrt(rrIntervals.reduce((s, rr) => s + (rr - meanRR) ** 2, 0) / rrIntervals.length) : 0;
  const irregularity = meanRR > 0 ? stdRR / meanRR : 0;

  let rhythm: "regular" | "irregular" | "highly_irregular" = "regular";
  if (irregularity > 0.20) rhythm = "highly_irregular";
  else if (irregularity > 0.12) rhythm = "irregular";

  const displayLen = 200;
  const step = Math.max(1, Math.floor(smoothed.length / displayLen));
  const displaySignal: number[] = [];
  for (let i = 0; i < smoothed.length; i += step) displaySignal.push(smoothed[i]);

  return { hr, hrv, rrIntervals, irregularity, rhythm, quality, fs, signal: displaySignal };
}

// ============ Diagnostics history (localStorage) ============
// Every attempt (pulse or sound, success or failure) appends one line here so
// the user can copy the whole run history in one tap instead of screenshotting
// tiny on-screen text after every single attempt.
const DIAG_LOG_KEY = "cardiolisten_diag_log";
function pushDiagLog(line: string) {
  try {
    const now = new Date().toISOString().replace("T", " ").slice(0, 19);
    const existing: string[] = JSON.parse(localStorage.getItem(DIAG_LOG_KEY) || "[]");
    existing.push(`[${now}] ${line}`);
    while (existing.length > 20) existing.shift();
    localStorage.setItem(DIAG_LOG_KEY, JSON.stringify(existing));
  } catch {}
}
function readDiagLog(): string[] {
  try { return JSON.parse(localStorage.getItem(DIAG_LOG_KEY) || "[]"); } catch { return []; }
}

// ============ Live Waveform Component ============

function LiveWaveform({ data, color, height = 80 }: { data: number[]; color: string; height?: number }) {
  if (data.length < 2) return <div className="w-full bg-slate-700/30 rounded-xl" style={{ height }} />;

  const visible = data.slice(-WAVEFORM_POINTS);
  const min = Math.min(...visible);
  const max = Math.max(...visible);
  const range = max - min || 1;

  const points = visible.map((v, i) => {
    const x = (i / (WAVEFORM_POINTS - 1)) * 100;
    const y = 90 - ((v - min) / range) * 80;
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="w-full bg-slate-900/50 rounded-xl p-2 border border-slate-700/50" style={{ height }}>
      <svg viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="none">
        {[25, 50, 75].map(y => (
          <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="#334155" strokeWidth="0.3" />
        ))}
        <polyline fill="none" stroke={color} strokeWidth="0.8" points={points} />
      </svg>
    </div>
  );
}

// ============ MAIN COMPONENT ============

export default function Home() {
  const [tab, setTab] = useState<Tab>("pulse");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pulse state
  const [pulseState, setPulseState] = useState<PulseState>("idle");
  const [pulseResult, setPulseResult] = useState<PulseResult | null>(null);
  const [pulseCountdown, setPulseCountdown] = useState(PPG_DURATION);
  const [fingerDetected, setFingerDetected] = useState(false);
  const [torchOn, setTorchOnState] = useState(false);
  const [liveHR, setLiveHR] = useState<number | null>(null);
  const [ppgWaveform, setPpgWaveform] = useState<number[]>([]);
  const [pulseDiag, setPulseDiagState] = useState<PulseDiag | null>(null);
  // Mirrors of the above so useCallback([]) closures (stopPPG etc.) can read
  // the live value instead of the stale one captured at first render.
  const torchOnRef = useRef(false);
  const pulseDiagRef = useRef<PulseDiag | null>(null);
  const setTorchOn = useCallback((v: boolean) => { torchOnRef.current = v; setTorchOnState(v); }, []);
  const setPulseDiag = useCallback((updater: PulseDiag | null | ((d: PulseDiag | null) => PulseDiag | null)) => {
    setPulseDiagState(prev => {
      const next = typeof updater === "function" ? (updater as (d: PulseDiag | null) => PulseDiag | null)(prev) : updater;
      pulseDiagRef.current = next;
      return next;
    });
  }, []);

  // Sound state
  const [soundState, setSoundState] = useState<SoundState>("idle");
  const [soundResult, setSoundResult] = useState<SoundResult | null>(null);
  const [soundCountdown, setSoundCountdown] = useState(RECORD_DURATION);
  const [soundWaveform, setSoundWaveform] = useState<number[]>([]);
  const [soundDiag, setSoundDiag] = useState<SoundDiag | null>(null);
  const [soundDebug, setSoundDebug] = useState<string | null>(null);

  // Refs
  const modelRef = useRef<tf.LayersModel | null>(null);
  const normRef = useRef<{ mean: number[]; std: number[] } | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ppgValsRef = useRef<number[]>([]);
  const ppgTimesRef = useRef<number[]>([]);
  const ppgFilteredRef = useRef<number[]>([]);
  const ppgTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ppgRafRef = useRef<number>(0);
  const ppgStreamRef = useRef<MediaStream | null>(null);
  const ppgStartRef = useRef<number>(0);
  const lastLiveHRRef = useRef<number>(0);
  const ppgTrackRef = useRef<MediaStreamTrack | null>(null);
  const exposureLockAttemptedRef = useRef(false);
  const stableFingerSinceRef = useRef<number | null>(null);
  const diagLastUpdateRef = useRef(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const soundTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundAnimRef = useRef<number>(0);
  const soundCtxRef = useRef<AudioContext | null>(null);
  const soundLevelPeakRef = useRef(0);

  // Load model
  useEffect(() => {
    tf.loadLayersModel("/model/model.json").then(model => {
      modelRef.current = model;
      fetch("/model/norm_params.json").then(r => r.json()).then(norm => {
        normRef.current = norm;
        setModelLoaded(true);
      });
    }).catch(e => console.error("Model load error:", e));
  }, []);

  // ============ PPG LOGIC ============
  const startPPG = useCallback(async () => {
    setError(null);
    setPulseResult(null);
    ppgValsRef.current = [];
    ppgTimesRef.current = [];
    ppgFilteredRef.current = [];
    setPpgWaveform([]);
    setLiveHR(null);
    setTorchOn(false);
    setPulseDiag(null);
    lastLiveHRRef.current = 0;
    exposureLockAttemptedRef.current = false;
    stableFingerSinceRef.current = null;

    try {
      // Prefer a real rear camera (torch usually only exists on the back camera)
      let deviceId: string | undefined;
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === "videoinput");
        const back = cams.find(c => /back|rear|environment/i.test(c.label));
        if (back) deviceId = back.deviceId;
      } catch {}

      const constraints: MediaStreamConstraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 60 } }
          : { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 60 } },
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      ppgStreamRef.current = stream;

      // Engage the torch/flash and verify it actually turned on. Torch control
      // via the web platform is known to be unreliable on some Android
      // devices even when getCapabilities() reports it — verify via
      // getSettings() rather than trusting the applyConstraints() resolve.
      const track = stream.getVideoTracks()[0];
      ppgTrackRef.current = track;
      const caps = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & {
        torch?: boolean; exposureMode?: string[]; whiteBalanceMode?: string[];
      };
      const exposureCapable = Array.isArray(caps.exposureMode) && caps.exposureMode.includes("manual");
      if (caps.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true } as MediaTrackConstraintSet] });
          const settings = (track.getSettings?.() ?? {}) as MediaTrackSettings & { torch?: boolean };
          setTorchOn(settings.torch === true);
        } catch { setTorchOn(false); }
      } else {
        setTorchOn(false);
      }
      setPulseDiag({ torchCapable: !!caps.torch, exposureCapable, exposureLockState: exposureCapable ? "pending" : "n/a", r: 0, g: 0, b: 0, fps: 0, ampPP: 0 });

      setPulseState("measuring");
      await new Promise(r => setTimeout(r, 150));

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try { await videoRef.current.play(); } catch {}
      }

      setPulseCountdown(PPG_DURATION);
      ppgStartRef.current = performance.now();

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

      // Read one frame, timestamped with the frame's true media time when available.
      const readFrame = (tsMs: number) => {
        const video = videoRef.current;
        if (video && video.readyState >= 2) {
          canvas.width = video.videoWidth || 320;
          canvas.height = video.videoHeight || 240;
          ctx.drawImage(video, 0, 0);
          const cx = Math.floor(canvas.width / 2);
          const cy = Math.floor(canvas.height / 2);
          const size = Math.min(60, Math.floor(canvas.width / 4));
          const imageData = ctx.getImageData(Math.max(0, cx - size), Math.max(0, cy - size), size * 2, size * 2);
          const d = imageData.data;
          let rSum = 0, gSum = 0, bSum = 0, count = 0;
          for (let i = 0; i < d.length; i += 4) { rSum += d[i]; gSum += d[i + 1]; bSum += d[i + 2]; count++; }
          const rAvg = rSum / count, gAvg = gSum / count, bAvg = bSum / count;

          // Finger over a flash-lit lens => red dominates and is bright.
          // Without a working flash the frame is much dimmer, so also accept
          // a looser "red-dominant" match at lower brightness (ambient light
          // through the fingertip) rather than only the flash-lit case.
          const isFinger = rAvg > gAvg * 1.15 && rAvg > bAvg * 1.15 && rAvg > 40;
          setFingerDetected(isFinger);

          ppgValsRef.current.push(rAvg);
          ppgTimesRef.current.push(tsMs);

          // real-time high-pass for the display waveform
          const vals = ppgValsRef.current;
          if (vals.length > 8) {
            const winSize = Math.min(vals.length, 45);
            const recent = vals.slice(-winSize);
            const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
            ppgFilteredRef.current.push(rAvg - mean);
            setPpgWaveform([...ppgFilteredRef.current]);
          }

          // Once the finger signal looks stable and bright for ~1.2s, lock
          // exposure so the phone's auto-exposure stops fighting the pulse
          // signal (this is the single biggest known cause of bad
          // camera-PPG accuracy on Android). Attempted once per session.
          const now = performance.now();
          if (isFinger) {
            if (stableFingerSinceRef.current === null) stableFingerSinceRef.current = now;
          } else {
            stableFingerSinceRef.current = null;
          }
          if (!exposureLockAttemptedRef.current && stableFingerSinceRef.current !== null &&
              now - stableFingerSinceRef.current > 1200 && ppgTrackRef.current) {
            exposureLockAttemptedRef.current = true;
            const t = ppgTrackRef.current;
            (async () => {
              try {
                // exposureMode and exposureTime must be applied in separate
                // calls — the mode switch has to land before a time can stick.
                await t.applyConstraints({ advanced: [{ exposureMode: "manual" } as MediaTrackConstraintSet] });
                const settings = (t.getSettings?.() ?? {}) as MediaTrackSettings & { exposureMode?: string };
                const ok = settings.exposureMode === "manual";
                setPulseDiag(d => d ? { ...d, exposureLockState: ok ? "locked" : "failed" } : d);
              } catch {
                setPulseDiag(d => d ? { ...d, exposureLockState: "failed" } : d);
              }
            })();
          }

          // Throttled diagnostics readout (~4/s) so the on-screen panel is
          // legible instead of refreshing every camera frame.
          if (now - diagLastUpdateRef.current > 250) {
            diagLastUpdateRef.current = now;
            const times = ppgTimesRef.current;
            let fps = 0;
            if (times.length > 10) {
              const recentT = times.slice(-30);
              const span = recentT[recentT.length - 1] - recentT[0];
              fps = span > 0 ? ((recentT.length - 1) * 1000) / span : 0;
            }
            const recentFiltered = ppgFilteredRef.current.slice(-60);
            const ampPP = recentFiltered.length > 1 ? Math.max(...recentFiltered) - Math.min(...recentFiltered) : 0;
            setPulseDiag(d => d ? { ...d, r: Math.round(rAvg), g: Math.round(gAvg), b: Math.round(bAvg), fps, ampPP } : d);
          }
        }
      };

      // Use requestVideoFrameCallback when available: fires exactly once per
      // presented camera frame with an accurate media timestamp.
      const vAny = videoRef.current as unknown as {
        requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
      };
      if (vAny && typeof vAny.requestVideoFrameCallback === "function") {
        const onFrame = (_now: number, meta: { mediaTime: number }) => {
          readFrame(meta.mediaTime * 1000);
          ppgRafRef.current = vAny.requestVideoFrameCallback!(onFrame);
        };
        ppgRafRef.current = vAny.requestVideoFrameCallback!(onFrame);
      } else {
        // Fallback: rAF with performance.now(); dedupe identical timestamps.
        const loop = () => {
          readFrame(performance.now() - ppgStartRef.current);
          ppgRafRef.current = requestAnimationFrame(loop);
        };
        loop();
      }

      // Live HR update every ~2s using the last 10s of timestamped samples
      let remaining = PPG_DURATION;
      ppgTimerRef.current = setInterval(() => {
        remaining--;
        setPulseCountdown(remaining);

        const vals = ppgValsRef.current, times = ppgTimesRef.current;
        if (vals.length > 40 && times.length === vals.length) {
          const cutoff = times[times.length - 1] - 10000;
          let s = 0; while (s < times.length && times[s] < cutoff) s++;
          const wv = vals.slice(s), wt = times.slice(s);
          try {
            const { signal, fs } = resampleUniform(wv, wt);
            const detr = detrendSignal(signal, fs);
            const sm = smoothSignal(detr, Math.max(2, Math.round(fs / 8)));
            const { bpm } = fftHeartRate(sm, fs);
            if (bpm >= 40 && bpm <= 200) { setLiveHR(bpm); lastLiveHRRef.current = bpm; }
          } catch {}
        }
        if (remaining <= 0) stopPPG();
      }, 1000);
    } catch {
      setError("Camera access denied or unavailable. Allow camera access and use a phone with a rear camera + flash.");
      setPulseState("idle");
    }
  }, []);

  const stopPPG = useCallback(() => {
    if (ppgTimerRef.current) clearInterval(ppgTimerRef.current);
    const vAny = videoRef.current as unknown as { cancelVideoFrameCallback?: (id: number) => void };
    if (vAny?.cancelVideoFrameCallback) vAny.cancelVideoFrameCallback(ppgRafRef.current);
    cancelAnimationFrame(ppgRafRef.current);
    if (ppgStreamRef.current) ppgStreamRef.current.getTracks().forEach(t => t.stop());
    ppgTrackRef.current = null;

    setPulseState("processing");
    const vals = ppgValsRef.current, times = ppgTimesRef.current;
    const d = pulseDiagRef.current;
    const diagTail = d ? `torch:${d.torchCapable ? (torchOnRef.current ? "on" : "off") : "n/a"} explock:${d.exposureLockState} R:${d.r} G:${d.g} B:${d.b} fps:${d.fps.toFixed(1)} amp:${d.ampPP.toFixed(1)}` : "no diag captured";
    if (vals.length < 60) {
      pushDiagLog(`PULSE FAIL: not enough frames (${vals.length}) | ${diagTail}`);
      setError("Not enough data. Keep your finger over the camera for the full reading.");
      setPulseState("idle");
      return;
    }
    try {
      const result = analyzePPG(vals, times);
      if (result.hr < 35 || result.hr > 210 || result.quality < 0.08) {
        pushDiagLog(`PULSE FAIL: hr=${result.hr} quality=${result.quality.toFixed(2)} | ${diagTail}`);
        setError("Couldn't detect a clean pulse. Cover the rear camera + flash fully with your fingertip, hold still, and try again.");
        setPulseState("idle");
        return;
      }
      pushDiagLog(`PULSE OK: hr=${result.hr} hrv=${result.hrv} quality=${result.quality.toFixed(2)} | ${diagTail}`);
      setPulseResult(result);
      setPulseState("done");
    } catch (e) {
      pushDiagLog(`PULSE ERROR: ${e instanceof Error ? e.message : String(e)} | ${diagTail}`);
      setError("Analysis failed. Try again.");
      setPulseState("idle");
    }
  }, []);

  // ============ SOUND LOGIC ============
  const startSound = useCallback(async () => {
    setError(null);
    setSoundResult(null);
    setSoundDiag(null);
    setSoundDebug(null);
    audioChunksRef.current = [];
    setSoundWaveform([]);
    soundLevelPeakRef.current = 0;

    try {
      // Record RAW mic audio (no AGC/NS/EC) so features match the training
      // pipeline (librosa.load at 2 kHz, no bandpass/gain).
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      audioStreamRef.current = stream;

      // Analyser purely for the live waveform display
      const audioCtx = new AudioContext();
      soundCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm",
      });
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        await processSound();
      };

      const waveData = new Float32Array(analyser.fftSize);
      let lastDiagUpdate = 0;
      const updateWaveform = () => {
        analyser.getFloatTimeDomainData(waveData);
        const step = Math.floor(waveData.length / 50);
        const points: number[] = [];
        for (let i = 0; i < waveData.length; i += step) points.push(waveData[i]);
        setSoundWaveform(prev => [...prev, ...points].slice(-WAVEFORM_POINTS * 2));

        // Live mic level meter (RMS + running peak) so a too-quiet or
        // clipping recording is visible on screen instead of discovered
        // after the fact from a failed classification.
        const now = performance.now();
        if (now - lastDiagUpdate > 200) {
          lastDiagUpdate = now;
          let sumSq = 0, peak = 0;
          for (let i = 0; i < waveData.length; i++) {
            const v = waveData[i];
            sumSq += v * v;
            if (Math.abs(v) > peak) peak = Math.abs(v);
          }
          const rms = Math.sqrt(sumSq / waveData.length);
          soundLevelPeakRef.current = Math.max(soundLevelPeakRef.current * 0.98, peak);
          setSoundDiag({ level: rms, peak: soundLevelPeakRef.current });
        }
        soundAnimRef.current = requestAnimationFrame(updateWaveform);
      };
      updateWaveform();

      mediaRecorder.start(100);
      setSoundState("recording");
      setSoundCountdown(RECORD_DURATION);

      let remaining = RECORD_DURATION;
      soundTimerRef.current = setInterval(() => {
        remaining--;
        setSoundCountdown(remaining);
        if (remaining <= 0) stopSound();
      }, 1000);
    } catch {
      setError("Microphone access denied.");
      setSoundState("idle");
    }
  }, []);

  const stopSound = useCallback(() => {
    if (soundTimerRef.current) clearInterval(soundTimerRef.current);
    cancelAnimationFrame(soundAnimRef.current);
    if (soundCtxRef.current) { soundCtxRef.current.close(); soundCtxRef.current = null; }
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
  }, []);

  const processSound = useCallback(async () => {
    setSoundState("processing");
    try {
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();

      const rawData = audioBuffer.getChannelData(0);
      // Anti-aliased resample to the model's 2 kHz (matches librosa.load(sr=2000))
      const resampled = resampleAudio(rawData, audioBuffer.sampleRate, SAMPLE_RATE);

      let rawMax = 0, rawSumSq = 0;
      for (let i = 0; i < rawData.length; i++) { const a = Math.abs(rawData[i]); if (a > rawMax) rawMax = a; rawSumSq += rawData[i] * rawData[i]; }
      const rawRms = Math.sqrt(rawSumSq / rawData.length);
      const debugBase = `recorded ${(audioBuffer.length / audioBuffer.sampleRate).toFixed(1)}s @ ${audioBuffer.sampleRate}Hz, ` +
        `mic level: peak ${(rawMax * 100).toFixed(0)}% / rms ${(rawRms * 100).toFixed(1)}%`;

      const { windows: windowScores, totalWindows } = scoreAudioWindows(resampled, SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_DURATION / 2);

      if (windowScores.length === 0) {
        const msg = `${debugBase}, ${totalWindows} windows all below silence threshold (rms<0.0005) - mic likely picked up almost nothing`;
        setSoundDebug(msg);
        pushDiagLog(`SOUND FAIL: ${msg}`);
        setError("No usable audio detected. Record again in a quiet room with the mic pressed to your chest.");
        setSoundState("idle");
        return;
      }

      const sortedWindows = windowScores.slice().sort((a, b) => b.score - a.score);
      const bestWindows = sortedWindows.slice(0, Math.min(5, sortedWindows.length));
      const norm = normRef.current!;
      const windowSize = SAMPLE_RATE * SEGMENT_DURATION;
      const segmentResults: { normal: number; abnormal: number; score: number }[] = [];

      for (const win of bestWindows) {
        const segment = resampled.slice(win.start, win.start + windowSize);
        const mfccFrames = extractMfcc(segment);
        const paddedFrames: Float64Array[] = [];
        for (let i = 0; i < MAX_FRAMES; i++) paddedFrames.push(i < mfccFrames.length ? mfccFrames[i] : new Float64Array(N_MFCC));

        const inputData = new Float32Array(MAX_FRAMES * N_MFCC);
        for (let i = 0; i < MAX_FRAMES; i++)
          for (let j = 0; j < N_MFCC; j++)
            inputData[i * N_MFCC + j] = (paddedFrames[i][j] - norm.mean[j]) / (norm.std[j] + 1e-8);

        const inputTensor = tf.tensor3d(inputData, [1, MAX_FRAMES, N_MFCC]);
        const prediction = modelRef.current!.predict(inputTensor) as tf.Tensor;
        const probs = await prediction.data();
        inputTensor.dispose();
        prediction.dispose();
        segmentResults.push({ normal: probs[0], abnormal: probs[1], score: win.score });
      }

      let totalWeight = 0, wNormal = 0, wAbnormal = 0;
      for (const r of segmentResults) {
        wNormal += r.normal * r.score;
        wAbnormal += r.abnormal * r.score;
        totalWeight += r.score;
      }
      const avgNormal = wNormal / (totalWeight || 1);
      const avgAbnormal = wAbnormal / (totalWeight || 1);
      const bestQuality = bestWindows[0]?.score ?? 0;

      setSoundResult({
        label: avgAbnormal > avgNormal ? "abnormal" : "normal",
        confidence: Math.max(avgNormal, avgAbnormal),
        normal: avgNormal,
        abnormal: avgAbnormal,
        segmentsAnalyzed: segmentResults.length,
        quality: bestQuality,
      });
      const okMsg = `${debugBase}, ${windowScores.length}/${totalWindows} windows usable, best window rms ${(bestWindows[0]?.rms * 100 || 0).toFixed(2)}%`;
      setSoundDebug(okMsg);
      pushDiagLog(`SOUND OK: label=${avgAbnormal > avgNormal ? "abnormal" : "normal"} conf=${Math.max(avgNormal, avgAbnormal).toFixed(2)} | ${okMsg}`);
      setSoundState("done");
    } catch (e) {
      console.error(e);
      const msg = "Recording failed to decode - check microphone permission and try again.";
      setSoundDebug(prev => prev ?? msg);
      pushDiagLog(`SOUND ERROR: ${e instanceof Error ? e.message : String(e)}`);
      setError("Audio processing failed.");
      setSoundState("idle");
    }
  }, []);

  // ============ DIAGNOSTICS COPY ============
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const [diagCount, setDiagCount] = useState(0);
  useEffect(() => { setDiagCount(readDiagLog().length); }, [pulseState, soundState]);
  const copyDiagnostics = useCallback(async () => {
    const lines = readDiagLog();
    const text = lines.length > 0 ? lines.join("\n") : "No attempts logged yet.";
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus("copied");
    } catch {
      setCopyStatus("failed");
    }
    setTimeout(() => setCopyStatus("idle"), 2500);
  }, []);

  // ============ RENDER ============
  const tabClass = (t: Tab) => `flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${tab === t ? "bg-slate-700 text-white" : "text-slate-400 hover:text-slate-300"}`;

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-white p-4">
      <div className="mt-4 mb-3 text-center">
        <h1 className="text-2xl font-bold tracking-tight">CardioListen</h1>
        <p className="text-slate-400 text-xs mt-0.5">Heart screening &mdash; not a medical diagnosis</p>
      </div>

      <div className="w-full max-w-sm flex gap-1 bg-slate-800/80 p-1 rounded-xl mb-2">
        <button className={tabClass("pulse")} onClick={() => setTab("pulse")}>Pulse</button>
        <button className={tabClass("sound")} onClick={() => setTab("sound")}>Heart Sound</button>
        <button className={tabClass("report")} onClick={() => setTab("report")}>Report</button>
      </div>

      {diagCount > 0 && (
        <button onClick={copyDiagnostics} className="w-full max-w-sm mb-3 py-1.5 px-3 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg text-[11px] text-slate-400 flex items-center justify-center gap-1.5 transition-colors">
          {copyStatus === "copied" ? "Copied! Paste it in the chat." : copyStatus === "failed" ? "Copy failed - long-press to select instead" : `Copy diagnostics from last ${diagCount} attempt${diagCount === 1 ? "" : "s"}`}
        </button>
      )}

      <div className="w-full max-w-sm bg-slate-800/50 rounded-2xl border border-slate-700 p-4 backdrop-blur">

        {/* ============ PULSE TAB ============ */}
        {tab === "pulse" && (
          <>
            {pulseState === "idle" && (
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-rose-500/10 border-2 border-rose-500/30 flex items-center justify-center">
                  <svg className="w-8 h-8 text-rose-400" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-slate-300 text-sm font-medium">Fingertip Pulse Reading</p>
                  <p className="text-slate-500 text-xs mt-1">Use a phone. Cover the rear camera + flash fully with your fingertip and hold still.</p>
                </div>
                <button onClick={startPPG} className="w-full py-3 bg-rose-500 hover:bg-rose-600 rounded-xl font-semibold transition-colors">
                  Start Pulse Reading
                </button>
              </div>
            )}

            {pulseState === "measuring" && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-3">
                  <div className="relative flex-shrink-0">
                    <video ref={videoRef} className="w-16 h-16 rounded-full object-cover border-2 border-rose-500" muted playsInline autoPlay />
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                  <div className="flex-1">
                    <div className={`text-xs font-medium ${fingerDetected ? "text-green-400" : "text-yellow-400 animate-pulse"}`}>
                      {fingerDetected ? "Finger detected" : "Cover the rear camera + flash"}
                    </div>
                    {!torchOn && <div className="text-[10px] text-amber-400/80 mt-0.5">Flash off &mdash; use bright light or a phone with flash</div>}
                    <div className="flex items-baseline gap-2 mt-1">
                      {liveHR && <span className="text-3xl font-bold text-rose-400">{liveHR}</span>}
                      {liveHR && <span className="text-xs text-slate-400">BPM</span>}
                      {!liveHR && <span className="text-sm text-slate-500">Detecting...</span>}
                    </div>
                  </div>
                  <div className="w-12 h-12 rounded-full border-2 border-rose-500/50 flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-bold">{pulseCountdown}</span>
                  </div>
                </div>

                <LiveWaveform data={ppgWaveform} color="#f43f5e" height={120} />

                <div className="w-full bg-slate-700 rounded-full h-1">
                  <div className="bg-rose-500 h-1 rounded-full transition-all" style={{ width: `${((PPG_DURATION - pulseCountdown) / PPG_DURATION) * 100}%` }} />
                </div>

                {pulseDiag && (
                  <div className="bg-slate-900/60 rounded-lg p-2 border border-slate-700/50 text-[10px] text-slate-400 font-mono leading-relaxed">
                    <div>torch: {pulseDiag.torchCapable ? (torchOn ? "on ✓" : "capable but NOT on ✗") : "not supported on this camera ✗"}</div>
                    <div>exposure lock: {pulseDiag.exposureLockState === "n/a" ? "not supported ✗" : pulseDiag.exposureLockState === "locked" ? "locked ✓" : pulseDiag.exposureLockState === "failed" ? "attempted, failed ✗" : "waiting for stable finger..."}</div>
                    <div>R:{pulseDiag.r} G:{pulseDiag.g} B:{pulseDiag.b} &middot; fps:{pulseDiag.fps.toFixed(1)} &middot; signal amp:{pulseDiag.ampPP.toFixed(1)}</div>
                  </div>
                )}

                <button onClick={stopPPG} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Stop &amp; Analyze
                </button>
              </div>
            )}

            {pulseState === "processing" && (
              <div className="flex flex-col items-center gap-4 py-8">
                <div className="w-12 h-12 border-4 border-rose-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-rose-400 text-sm">Analyzing pulse data...</p>
              </div>
            )}

            {pulseState === "done" && pulseResult && (
              <div className="flex flex-col gap-3">
                <div className="text-center">
                  <div className="text-5xl font-bold text-rose-400">{pulseResult.hr}</div>
                  <div className="text-slate-400 text-sm">BPM</div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    Signal quality: {pulseResult.quality > 0.4 ? "good" : pulseResult.quality > 0.2 ? "fair" : "low"} &middot; {Math.round(pulseResult.fs)} fps
                  </div>
                  {pulseDiag && (
                    <div className="text-[9px] text-slate-600 font-mono mt-1">
                      torch:{pulseDiag.torchCapable ? (torchOn ? "on" : "off✗") : "n/a"} exp-lock:{pulseDiag.exposureLockState} R:{pulseDiag.r} G:{pulseDiag.g} B:{pulseDiag.b}
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-slate-700/50 rounded-xl p-3 text-center">
                    <div className="text-lg font-bold text-blue-400">{pulseResult.hrv}</div>
                    <div className="text-xs text-slate-500">HRV (ms)</div>
                  </div>
                  <div className="bg-slate-700/50 rounded-xl p-3 text-center">
                    <div className={`text-lg font-bold ${pulseResult.rhythm === "regular" ? "text-green-400" : pulseResult.rhythm === "irregular" ? "text-yellow-400" : "text-red-400"}`}>
                      {pulseResult.rhythm === "regular" ? "Regular" : pulseResult.rhythm === "irregular" ? "Irregular" : "Very Irregular"}
                    </div>
                    <div className="text-xs text-slate-500">Rhythm</div>
                  </div>
                </div>
                <div className="bg-slate-900/50 rounded-xl p-2 border border-slate-700/50">
                  <div className="text-xs text-slate-500 mb-1">Pulse Waveform</div>
                  <svg viewBox={`0 0 ${pulseResult.signal.length} 100`} className="w-full h-16" preserveAspectRatio="none">
                    <polyline fill="none" stroke="#f43f5e" strokeWidth="1.5"
                      points={pulseResult.signal.map((v, i) => {
                        const min = Math.min(...pulseResult.signal);
                        const max = Math.max(...pulseResult.signal);
                        const y = max === min ? 50 : 90 - ((v - min) / (max - min)) * 80;
                        return `${i},${y}`;
                      }).join(" ")} />
                  </svg>
                </div>
                <div className="space-y-1 text-xs text-slate-400">
                  <p>HR {pulseResult.hr >= 60 && pulseResult.hr <= 100 ? "is within" : pulseResult.hr < 60 ? "is below" : "is above"} the typical resting range (60-100)</p>
                  <p>HRV {pulseResult.hrv > 50 ? "suggests good" : pulseResult.hrv > 20 ? "is moderate" : "is low"} short-term variability</p>
                  {pulseResult.rhythm !== "regular" && <p className="text-yellow-400">Irregular rhythm detected &mdash; if this repeats, see a clinician</p>}
                </div>
                <button onClick={() => { setPulseState("idle"); setPulseResult(null); setLiveHR(null); setPpgWaveform([]); }} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Measure Again
                </button>
              </div>
            )}
          </>
        )}

        {/* ============ SOUND TAB ============ */}
        {tab === "sound" && (
          <>
            {soundState === "idle" && (
              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-blue-500/10 border-2 border-blue-500/30 flex items-center justify-center">
                  <svg className="w-8 h-8 text-blue-400" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-slate-300 text-sm font-medium">Heart Sound Screening</p>
                  <p className="text-slate-500 text-xs mt-1">Press the phone mic firmly on your bare left chest in a quiet room.</p>
                  <p className="text-amber-400/80 text-[11px] mt-1">Experimental. A phone mic is not a stethoscope &mdash; treat results as a rough screen only.</p>
                </div>
                {!modelLoaded && <p className="text-yellow-400 text-xs animate-pulse">Loading AI model...</p>}
                <button onClick={startSound} disabled={!modelLoaded} className="w-full py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-600 rounded-xl font-semibold transition-colors">
                  Start Recording
                </button>
              </div>
            )}

            {soundState === "recording" && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-400 font-medium text-sm animate-pulse">Recording Heart Sound</p>
                    <p className="text-slate-500 text-xs">Hold still, keep the room quiet</p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-lg font-bold">{soundCountdown}</span>
                  </div>
                </div>

                <LiveWaveform data={soundWaveform} color="#3b82f6" height={140} />

                {soundDiag && (
                  <div className="bg-slate-900/60 rounded-lg p-2 border border-slate-700/50">
                    <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                      <span>mic level</span>
                      <span className={soundDiag.peak > 0.9 ? "text-red-400" : soundDiag.peak < 0.02 ? "text-amber-400" : "text-green-400"}>
                        {soundDiag.peak > 0.9 ? "clipping ✗" : soundDiag.peak < 0.02 ? "too quiet ✗" : "OK ✓"} (peak {(soundDiag.peak * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${soundDiag.peak > 0.9 ? "bg-red-500" : "bg-blue-500"}`} style={{ width: `${Math.min(100, soundDiag.peak * 100)}%` }} />
                    </div>
                  </div>
                )}

                <div className="w-full bg-slate-700 rounded-full h-1">
                  <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${((RECORD_DURATION - soundCountdown) / RECORD_DURATION) * 100}%` }} />
                </div>

                <button onClick={() => stopSound()} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Stop &amp; Analyze
                </button>
              </div>
            )}

            {soundState === "processing" && (
              <div className="flex flex-col items-center gap-4 py-8">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-blue-400 text-sm">Analyzing best segments...</p>
                <p className="text-slate-500 text-xs">Selecting the cleanest windows for inference</p>
              </div>
            )}

            {soundState === "done" && soundResult && (
              <div className="flex flex-col items-center gap-3">
                <div className={`w-14 h-14 rounded-full flex items-center justify-center ${soundResult.label === "normal" ? "bg-green-500/10 border-2 border-green-500" : "bg-yellow-500/10 border-2 border-yellow-500"}`}>
                  {soundResult.label === "normal" ? (
                    <svg className="w-7 h-7 text-green-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                  ) : (
                    <svg className="w-7 h-7 text-yellow-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" /></svg>
                  )}
                </div>
                <div className="text-center">
                  <h2 className={`text-xl font-bold ${soundResult.label === "normal" ? "text-green-400" : "text-yellow-400"}`}>
                    {soundResult.label === "normal" ? "Sounds Normal" : "Possible Atypical Sound"}
                  </h2>
                  <p className="text-slate-400 text-xs mt-1">Confidence: {(soundResult.confidence * 100).toFixed(1)}%</p>
                  <p className="text-slate-500 text-xs">Best {soundResult.segmentsAnalyzed} segments &middot; recording quality {soundResult.quality > 0.5 ? "good" : soundResult.quality > 0.25 ? "fair" : "low"}</p>
                  {soundResult.quality < 0.25 && <p className="text-amber-400/80 text-[11px] mt-1">Low-quality recording &mdash; result may be unreliable, try again in a quiet room.</p>}
                  {soundDebug && <p className="text-slate-600 text-[9px] font-mono mt-1">{soundDebug}</p>}
                </div>
                <div className="w-full space-y-2">
                  <div>
                    <div className="flex justify-between text-xs mb-1"><span className="text-green-400">Normal</span><span className="text-slate-400">{(soundResult.normal * 100).toFixed(1)}%</span></div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden"><div className="h-full bg-green-500 rounded-full" style={{ width: `${soundResult.normal * 100}%` }} /></div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1"><span className="text-yellow-400">Atypical</span><span className="text-slate-400">{(soundResult.abnormal * 100).toFixed(1)}%</span></div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden"><div className="h-full bg-yellow-500 rounded-full" style={{ width: `${soundResult.abnormal * 100}%` }} /></div>
                  </div>
                </div>
                <p className="text-slate-500 text-[11px] text-center">This is a screening estimate, not a diagnosis. Any concern &mdash; see a clinician.</p>
                <button onClick={() => { setSoundState("idle"); setSoundResult(null); setSoundWaveform([]); }} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Record Again
                </button>
              </div>
            )}
          </>
        )}

        {/* ============ REPORT TAB ============ */}
        {tab === "report" && (
          <div className="flex flex-col gap-3">
            <h2 className="text-lg font-bold text-center">Health Report</h2>
            {!pulseResult && !soundResult ? (
              <div className="text-center py-6">
                <p className="text-slate-500 text-sm">Complete tests to generate report</p>
                <div className="flex gap-2 mt-3 justify-center">
                  <button onClick={() => setTab("pulse")} className="py-2 px-4 bg-rose-500/20 text-rose-400 rounded-lg text-xs">Pulse</button>
                  <button onClick={() => setTab("sound")} className="py-2 px-4 bg-blue-500/20 text-blue-400 rounded-lg text-xs">Sound</button>
                </div>
              </div>
            ) : (
              <>
                <div className="bg-slate-700/30 rounded-xl p-4 text-center">
                  <div className="text-xs text-slate-500 mb-1">Overall Screening</div>
                  {(() => {
                    let flags = 0, total = 0;
                    if (pulseResult) {
                      total += 3;
                      if (pulseResult.hr < 50 || pulseResult.hr > 110) flags++;
                      if (pulseResult.hrv < 20) flags++;
                      if (pulseResult.rhythm !== "regular") flags++;
                    }
                    if (soundResult) { total++; if (soundResult.label === "abnormal") flags++; }
                    const score = total > 0 ? Math.round(((total - flags) / total) * 100) : 0;
                    const color = score >= 80 ? "text-green-400" : score >= 50 ? "text-yellow-400" : "text-red-400";
                    const label = score >= 80 ? "No obvious flags" : score >= 50 ? "Some things to watch" : "Worth a check-up";
                    return (<><div className={`text-4xl font-bold ${color}`}>{score}%</div><div className={`text-sm font-medium ${color}`}>{label}</div></>);
                  })()}
                </div>
                <div className="space-y-2">
                  {pulseResult && (
                    <>
                      <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                        <span className="text-xs text-slate-400">Heart Rate</span>
                        <span className={`text-sm font-bold ${pulseResult.hr >= 60 && pulseResult.hr <= 100 ? "text-green-400" : "text-yellow-400"}`}>{pulseResult.hr} BPM</span>
                      </div>
                      <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                        <span className="text-xs text-slate-400">HRV</span>
                        <span className={`text-sm font-bold ${pulseResult.hrv > 50 ? "text-green-400" : pulseResult.hrv > 20 ? "text-yellow-400" : "text-red-400"}`}>{pulseResult.hrv}ms</span>
                      </div>
                      <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                        <span className="text-xs text-slate-400">Rhythm</span>
                        <span className={`text-sm font-bold ${pulseResult.rhythm === "regular" ? "text-green-400" : "text-yellow-400"}`}>{pulseResult.rhythm === "regular" ? "Regular" : "Irregular"}</span>
                      </div>
                    </>
                  )}
                  {soundResult && (
                    <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                      <span className="text-xs text-slate-400">Heart Sound</span>
                      <span className={`text-sm font-bold ${soundResult.label === "normal" ? "text-green-400" : "text-yellow-400"}`}>{soundResult.label === "normal" ? "Normal" : "Atypical"} ({(soundResult.confidence * 100).toFixed(0)}%)</span>
                    </div>
                  )}
                </div>
                {(!pulseResult || !soundResult) && (
                  <div className="text-center">
                    <p className="text-slate-500 text-xs mb-1">Complete all tests for full report</p>
                    {!pulseResult && <button onClick={() => setTab("pulse")} className="py-1 px-3 bg-rose-500/20 text-rose-400 rounded-lg text-xs mr-2">+ Pulse</button>}
                    {!soundResult && <button onClick={() => setTab("sound")} className="py-1 px-3 bg-blue-500/20 text-blue-400 rounded-lg text-xs">+ Sound</button>}
                  </div>
                )}
                <p className="text-slate-500 text-[11px] text-center mt-1">Screening estimates only &mdash; not a diagnosis. Consult a clinician for any concern.</p>
              </>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm max-w-sm text-center">
          {error}
          <button onClick={() => setError(null)} className="ml-2 underline text-xs">dismiss</button>
          {pulseDiag && tab === "pulse" && (
            <div className="mt-2 pt-2 border-t border-red-500/20 text-[9px] text-red-300/70 font-mono text-left">
              torch:{pulseDiag.torchCapable ? (torchOn ? "on" : "off✗") : "n/a"} exp-lock:{pulseDiag.exposureLockState} R:{pulseDiag.r} G:{pulseDiag.g} B:{pulseDiag.b} fps:{pulseDiag.fps.toFixed(1)} amp:{pulseDiag.ampPP.toFixed(1)}
            </div>
          )}
          {soundDebug && tab === "sound" && (
            <div className="mt-2 pt-2 border-t border-red-500/20 text-[9px] text-red-300/70 font-mono text-left">{soundDebug}</div>
          )}
        </div>
      )}

      <div className="mt-4 max-w-sm text-center pb-6">
        <p className="text-slate-600 text-xs">For educational and wellness screening only. Not a medical device and not a diagnosis. If you have symptoms (chest pain, fainting, severe breathlessness) seek medical care immediately.</p>
      </div>
    </div>
  );
}
