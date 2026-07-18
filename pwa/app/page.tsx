"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// ============ CONSTANTS ============
// Audio / model (MUST match train_model.py + librosa defaults)
const SAMPLE_RATE = 2000;      // model was trained on librosa.load(sr=2000)
const SEGMENT_DURATION = 3;    // seconds
const CHECK_DURATION = 10;     // seconds to check each chest zone
const EXTEND_DURATION = 15;    // additional seconds recorded after a zone passes (25s total at that zone)
// Lowered from 0.5 to 0.18 after real-device testing (Pixel 6 Pro, 2026-07-18)
// showed genuine, model-usable recordings scoring ~0.23 -- the original 0.5
// was set without real calibration data and rejected every real attempt.
// See verify/window_score.mjs for the calibration against real device logs.
const GATE_PASS_THRESHOLD = 0.18; // scoreAudioWindow threshold for "clean enough to use"
const N_MFCC = 20;
const HOP_LENGTH = 128;
const N_FFT = 2048;            // librosa.feature.mfcc default n_fft
const N_MELS = 128;            // librosa.feature.melspectrogram default n_mels
const MAX_FRAMES = 47;         // 1 + floor(SAMPLE_RATE*SEGMENT_DURATION / HOP_LENGTH)
const TOP_DB = 80;             // librosa.power_to_db default

const WAVEFORM_POINTS = 300;

type Tab = "sound" | "report";
type SoundState = "idle" | "positioning" | "checking" | "recording" | "processing" | "done" | "all_zones_failed";
type ChestZone = "lower_left" | "upper_left" | "center";
const ZONE_ORDER: ChestZone[] = ["lower_left", "upper_left", "center"];
const ZONE_INFO: Record<ChestZone, { title: string; instruction: string }> = {
  lower_left: { title: "Lower-left chest", instruction: "Place the mic on your lower-left chest, near your ribs. This spot is usually clearest." },
  upper_left: { title: "Upper-left chest", instruction: "Place the mic on your upper-left chest, just below your collarbone." },
  center: { title: "Center chest", instruction: "Place the mic on the center of your chest, over your lower breastbone." },
};

interface SoundResult {
  label: string;
  confidence: number;
  normal: number;
  abnormal: number;
  segmentsAnalyzed: number;
  quality: number; // 0..1 best-window quality
}

// On-device diagnostics so real hardware failures can be diagnosed from a
// screenshot instead of guessed at.
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

  // rms >= 0.0005 already guaranteed by the early return above (the real
  // silence floor) -- only cap the UPPER end (clipping/too-loud). A second,
  // higher lower-bound here double-penalizes legitimately quiet recordings:
  // real phone-mic heart sounds average very low RMS over a 3s window since
  // most of the window is silence between S1/S2 beats. Confirmed against a
  // real device log (Pixel 6 Pro): a recording the model classified with 73%
  // confidence had best-window rms of only 0.06%, well below the old 0.1%
  // floor. See verify/window_score.mjs for the calibration.
  const rmsScore = rms < 0.5 ? 1 : 0.15;
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

// ============ Diagnostics history (localStorage) ============
// Every sound-flow attempt (success or failure) appends one line here so the
// user can copy the whole run history in one tap instead of screenshotting
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

// ============ Zone Map Component ============
// Abstract zone grid (approved over a body-silhouette illustration in
// brainstorming). Fixed visual layout — top-left, bottom-left, bottom-right —
// independent of attempt order; the active zone is highlighted regardless of
// where it sits in this fixed layout.

function ZoneMap({ activeZone }: { activeZone: ChestZone }) {
  const box = (zone: ChestZone, x: number, y: number, w: number, h: number) => {
    const active = zone === activeZone;
    return (
      <rect
        x={x} y={y} width={w} height={h} rx={12}
        fill={active ? "#3b82f6" : "#1e293b"}
        stroke={active ? "#60a5fa" : "#334155"}
        strokeWidth={active ? 2.5 : 1.5}
        opacity={active ? 0.9 : 0.6}
      />
    );
  };
  return (
    <svg viewBox="0 0 200 220" width="100%" height="180" className="bg-slate-900/50 rounded-xl">
      <rect x="30" y="20" width="140" height="180" rx="24" fill="none" stroke="#475569" strokeWidth="2" />
      <line x1="100" y1="20" x2="100" y2="200" stroke="#334155" strokeWidth="1" strokeDasharray="4 4" />
      {box("upper_left", 42, 35, 52, 52)}
      {box("lower_left", 42, 140, 52, 45)}
      {box("center", 106, 140, 52, 45)}
    </svg>
  );
}

// ============ MAIN COMPONENT ============

export default function Home() {
  const [tab, setTab] = useState<Tab>("sound");
  const [modelLoaded, setModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sound state
  const [soundState, setSoundState] = useState<SoundState>("idle");
  const [soundResult, setSoundResult] = useState<SoundResult | null>(null);
  const [soundCountdown, setSoundCountdown] = useState(EXTEND_DURATION);
  const [checkCountdown, setCheckCountdown] = useState(CHECK_DURATION);
  const [soundWaveform, setSoundWaveform] = useState<number[]>([]);
  const [soundDiag, setSoundDiag] = useState<SoundDiag | null>(null);
  const [soundDebug, setSoundDebug] = useState<string | null>(null);
  const [zoneMessage, setZoneMessage] = useState<string | null>(null);
  // zoneIndex is mirrored into zoneIndexRef so useCallback([]) closures read
  // the live zone instead of the value captured at first render.
  const [zoneIndex, setZoneIndexState] = useState(0);
  const zoneIndexRef = useRef(0);
  const setZoneIndex = useCallback((v: number) => { zoneIndexRef.current = v; setZoneIndexState(v); }, []);

  // Refs
  const modelRef = useRef<tf.LayersModel | null>(null);
  const normRef = useRef<{ mean: number[]; std: number[] } | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const soundTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundAnimRef = useRef<number>(0);
  const soundCtxRef = useRef<AudioContext | null>(null);
  const soundLevelPeakRef = useRef(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const beatStateRef = useRef<BeatEnvelopeState>(initBeatEnvelopeState());
  const zoneChunksRef = useRef<Partial<Record<ChestZone, Blob[]>>>({});
  const zoneScoresRef = useRef<Partial<Record<ChestZone, number>>>({});
  // Bumped whenever a guided-flow session starts or is cancelled, so async work
  // (evaluateZoneGate, analyzeZoneAudio) that resumes after an `await` can tell
  // it belongs to a stale session and bail out instead of driving state that no
  // longer applies (see stopGuidedFlow / startGuidedFlow).
  const guidedFlowEpochRef = useRef(0);

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

  // ============ SOUND LOGIC ============
  // ============ SOUND LOGIC (guided multi-zone flow) ============

  const teardownGuidedStream = useCallback(() => {
    if (audioStreamRef.current) { audioStreamRef.current.getTracks().forEach(t => t.stop()); audioStreamRef.current = null; }
    if (soundCtxRef.current) { soundCtxRef.current.close(); soundCtxRef.current = null; }
    analyserRef.current = null;
  }, []);

  const startGuidedFlow = useCallback(async () => {
    // New session: anything still in flight from a prior session (e.g. a
    // cancelled evaluateZoneGate/analyzeZoneAudio whose await just resolved)
    // is now stale and must not be allowed to drive this fresh session's state.
    guidedFlowEpochRef.current += 1;
    setError(null);
    setSoundResult(null);
    setSoundDiag(null);
    setSoundDebug(null);
    setZoneMessage(null);
    zoneChunksRef.current = {};
    zoneScoresRef.current = {};
    soundLevelPeakRef.current = 0;
    setZoneIndex(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      audioStreamRef.current = stream;

      const audioCtx = new AudioContext();
      soundCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      setSoundState("positioning");
    } catch {
      setError("Microphone access denied.");
      setSoundState("idle");
    }
  }, []);

  const stopGuidedFlow = useCallback(() => {
    // Invalidate any evaluateZoneGate/analyzeZoneAudio still awaiting in the
    // background so its eventual resolution becomes a no-op instead of
    // resurrecting state after this cancel.
    guidedFlowEpochRef.current += 1;
    if (soundTimerRef.current) clearInterval(soundTimerRef.current);
    cancelAnimationFrame(soundAnimRef.current);
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") { recorder.onstop = null; recorder.stop(); }
    teardownGuidedStream();
    setSoundState("idle");
    setZoneMessage(null);
  }, []);

  const analyzeZoneAudio = useCallback(async (chunks: Blob[], zone: ChestZone, epoch: number) => {
    // Guard against a stale call: this can be scheduled by a MediaRecorder
    // "onstop" handler (see finalizeZoneSuccess) that was armed before the
    // user cancelled, and the recorder's `state` flips to "inactive"
    // synchronously inside stop() — before the async stop event that runs the
    // handler — so stopGuidedFlow's `state === "recording"` check can race and
    // miss clearing it. Bail out before touching any state if that happened.
    if (epoch !== guidedFlowEpochRef.current) return;
    setSoundState("processing");
    try {
      const blob = new Blob(chunks, { type: "audio/webm" });
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();
      if (epoch !== guidedFlowEpochRef.current) return;

      const rawData = audioBuffer.getChannelData(0);
      const resampled = resampleAudio(rawData, audioBuffer.sampleRate, SAMPLE_RATE);

      let rawMax = 0, rawSumSq = 0;
      for (let i = 0; i < rawData.length; i++) { const a = Math.abs(rawData[i]); if (a > rawMax) rawMax = a; rawSumSq += rawData[i] * rawData[i]; }
      const rawRms = Math.sqrt(rawSumSq / rawData.length);
      const debugBase = `zone=${zone}, recorded ${(audioBuffer.length / audioBuffer.sampleRate).toFixed(1)}s @ ${audioBuffer.sampleRate}Hz, ` +
        `mic level: peak ${(rawMax * 100).toFixed(0)}% / rms ${(rawRms * 100).toFixed(1)}%`;

      const { windows: windowScores, totalWindows } = scoreAudioWindows(resampled, SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_DURATION / 2);

      if (windowScores.length === 0) {
        const msg = `${debugBase}, ${totalWindows} windows all below silence threshold (rms<0.0005) - mic likely picked up almost nothing`;
        setSoundDebug(msg);
        pushDiagLog(`SOUND FAIL: ${msg}`);
        setError("No usable audio detected. Let's try the guided placement again.");
        setSoundState("idle");
        teardownGuidedStream();
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
      if (epoch !== guidedFlowEpochRef.current) return;

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
      pushDiagLog(`SOUND OK: zone=${zone} label=${avgAbnormal > avgNormal ? "abnormal" : "normal"} conf=${Math.max(avgNormal, avgAbnormal).toFixed(2)} | ${okMsg}`);
      setSoundState("done");
      teardownGuidedStream();
    } catch (e) {
      if (epoch !== guidedFlowEpochRef.current) return;
      console.error(e);
      const msg = "Recording failed to decode - check microphone permission and try again.";
      setSoundDebug(prev => prev ?? msg);
      pushDiagLog(`SOUND ERROR: zone=${zone} ${e instanceof Error ? e.message : String(e)}`);
      setError("Audio processing failed.");
      setSoundState("idle");
      teardownGuidedStream();
    }
  }, []);

  const finalizeZoneSuccess = useCallback((zone: ChestZone) => {
    setZoneMessage(`Detected! Recording a bit more at your ${ZONE_INFO[zone].title.toLowerCase()}...`);
    setSoundState("recording");
    setCheckCountdown(0);
    setSoundCountdown(EXTEND_DURATION);

    let remaining = EXTEND_DURATION;
    soundTimerRef.current = setInterval(() => {
      remaining--;
      setSoundCountdown(remaining);
      if (remaining <= 0) {
        if (soundTimerRef.current) clearInterval(soundTimerRef.current);
        cancelAnimationFrame(soundAnimRef.current);
        const recorder = mediaRecorderRef.current;
        if (recorder && recorder.state === "recording") {
          // Capture the epoch right before stop(): recorder.state flips to
          // "inactive" synchronously inside stop(), so a cancel that lands in
          // the gap before the async "stop" event fires won't be caught by
          // stopGuidedFlow's `state === "recording"` check. analyzeZoneAudio
          // re-checks this epoch once its own awaits resolve.
          const epoch = guidedFlowEpochRef.current;
          recorder.onstop = () => { analyzeZoneAudio(audioChunksRef.current.slice(), zone, epoch); };
          recorder.stop();
        }
      }
    }, 1000);
  }, []);

  const advanceZone = useCallback(() => {
    cancelAnimationFrame(soundAnimRef.current);
    const recorder = mediaRecorderRef.current;
    const nextIndex = zoneIndexRef.current + 1;
    // Same recorder.stop()-vs-cancel race as finalizeZoneSuccess/analyzeZoneAudio:
    // capture the epoch now, since when goNext runs via the async "stop" event
    // it may be after this session was cancelled.
    const epoch = guidedFlowEpochRef.current;

    const goNext = () => {
      if (epoch !== guidedFlowEpochRef.current) return;
      if (nextIndex < ZONE_ORDER.length) {
        setZoneMessage("Didn't get a clean signal here.");
        setZoneIndex(nextIndex);
        setSoundState("positioning");
      } else {
        setSoundState("all_zones_failed");
      }
    };

    if (recorder && recorder.state === "recording") {
      recorder.onstop = goNext;
      recorder.stop();
    } else {
      goNext();
    }
  }, []);

  const evaluateZoneGate = useCallback(async () => {
    // Capture the session epoch before the await below. If the guided flow is
    // cancelled or restarted while decodeAudioData is in flight, stopGuidedFlow
    // / startGuidedFlow bump guidedFlowEpochRef, and the checks after the
    // await let this stale evaluation bail out instead of calling
    // finalizeZoneSuccess/advanceZone and resurrecting torn-down UI state.
    const epoch = guidedFlowEpochRef.current;
    const zone = ZONE_ORDER[zoneIndexRef.current];
    const chunksSoFar = audioChunksRef.current.slice();
    const blob = new Blob(chunksSoFar, { type: "audio/webm" });

    let score = 0;
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const scratchCtx = new AudioContext();
      const audioBuffer = await scratchCtx.decodeAudioData(arrayBuffer);
      scratchCtx.close();
      if (epoch !== guidedFlowEpochRef.current) return;
      const resampled = resampleAudio(audioBuffer.getChannelData(0), audioBuffer.sampleRate, SAMPLE_RATE);
      score = bestWindowScore(resampled, SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_DURATION / 2).score;
    } catch (e) {
      if (epoch !== guidedFlowEpochRef.current) return;
      // Decode failure is NOT the same thing as genuine silence (score=0 from
      // the try path) - log it distinctly so a real-device decode issue is
      // diagnosable and not confused with "the recorder just heard nothing".
      // Still counts as a FAIL for gating purposes (score stays 0), so it
      // falls into the same advanceZone() outcome as before - but through an
      // early return so exactly one diagnostic line is emitted per attempt.
      score = 0;
      zoneScoresRef.current[zone] = score;
      pushDiagLog(`SOUND ZONE CHECK: zone=${zone} DECODE_ERROR ${e instanceof Error ? e.message : String(e)}`);
      zoneChunksRef.current[zone] = chunksSoFar;
      advanceZone();
      return;
    }

    if (epoch !== guidedFlowEpochRef.current) return;

    zoneScoresRef.current[zone] = score;
    pushDiagLog(`SOUND ZONE CHECK: zone=${zone} score=${score.toFixed(2)} ${score > GATE_PASS_THRESHOLD ? "PASS" : "FAIL"}`);

    if (score > GATE_PASS_THRESHOLD) {
      finalizeZoneSuccess(zone);
    } else {
      zoneChunksRef.current[zone] = chunksSoFar;
      advanceZone();
    }
  }, []);

  const beginZoneCheck = useCallback(() => {
    const stream = audioStreamRef.current;
    const analyser = analyserRef.current;
    if (!stream || !analyser) return;

    setError(null);
    setZoneMessage(null);
    audioChunksRef.current = [];
    setSoundWaveform([]);
    beatStateRef.current = initBeatEnvelopeState();

    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm",
    });
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
    mediaRecorder.start(100);

    const waveData = new Float32Array(analyser.fftSize);
    let lastDiagUpdate = 0;
    const FRESH_TAIL = 512; // most-recent samples of the buffer only, avoids reprocessing the same audio every frame
    const updateLive = () => {
      analyser.getFloatTimeDomainData(waveData);

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

      let tailPeak = 0;
      for (let i = waveData.length - FRESH_TAIL; i < waveData.length; i++) {
        const a = Math.abs(waveData[i]);
        if (a > tailPeak) tailPeak = a;
      }
      const { state, beatDetected } = processBeatEnvelopeSample(beatStateRef.current, tailPeak, now);
      beatStateRef.current = state;
      setSoundWaveform(prev => [...prev, beatDetected ? 1 : 0].slice(-WAVEFORM_POINTS));

      soundAnimRef.current = requestAnimationFrame(updateLive);
    };
    updateLive();

    setSoundState("checking");
    setCheckCountdown(CHECK_DURATION);
    let remaining = CHECK_DURATION;
    soundTimerRef.current = setInterval(() => {
      remaining--;
      setCheckCountdown(remaining);
      if (remaining <= 0) {
        if (soundTimerRef.current) clearInterval(soundTimerRef.current);
        evaluateZoneGate();
      }
    }, 1000);
  }, []);

  const useBestZoneAnyway = useCallback(() => {
    const scores = zoneScoresRef.current;
    let bestZone: ChestZone | null = null, bestScore = -1;
    for (const z of ZONE_ORDER) {
      const s = scores[z] ?? -1;
      if (s > bestScore) { bestScore = s; bestZone = z; }
    }
    const chunks = bestZone ? zoneChunksRef.current[bestZone] : undefined;
    if (!bestZone || !chunks) {
      setError("No recording available. Please try again.");
      setSoundState("idle");
      teardownGuidedStream();
      return;
    }
    analyzeZoneAudio(chunks, bestZone, guidedFlowEpochRef.current);
  }, []);

  const retryGuidedFlow = useCallback(() => {
    zoneChunksRef.current = {};
    zoneScoresRef.current = {};
    setZoneIndex(0);
    setZoneMessage(null);
    setSoundState("positioning");
  }, []);

  // ============ DIAGNOSTICS COPY ============
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "failed">("idle");
  const [diagCount, setDiagCount] = useState(0);
  useEffect(() => { setDiagCount(readDiagLog().length); }, [soundState]);
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
        <button className={tabClass("sound")} onClick={() => setTab("sound")}>Heart Sound</button>
        <button className={tabClass("report")} onClick={() => setTab("report")}>Report</button>
      </div>

      {diagCount > 0 && (
        <button onClick={copyDiagnostics} className="w-full max-w-sm mb-3 py-1.5 px-3 bg-slate-800/60 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg text-[11px] text-slate-400 flex items-center justify-center gap-1.5 transition-colors">
          {copyStatus === "copied" ? "Copied! Paste it in the chat." : copyStatus === "failed" ? "Copy failed - long-press to select instead" : `Copy diagnostics from last ${diagCount} attempt${diagCount === 1 ? "" : "s"}`}
        </button>
      )}

      <div className="w-full max-w-sm bg-slate-800/50 rounded-2xl border border-slate-700 p-4 backdrop-blur">


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
                  <p className="text-slate-500 text-xs mt-1">We'll guide you to a few spots on your bare chest and check each one for a clear signal, in a quiet room.</p>
                  <p className="text-amber-400/80 text-[11px] mt-1">Experimental. A phone mic is not a stethoscope &mdash; treat results as a rough screen only.</p>
                </div>
                {!modelLoaded && <p className="text-yellow-400 text-xs animate-pulse">Loading AI model...</p>}
                <button onClick={startGuidedFlow} disabled={!modelLoaded} className="w-full py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-slate-600 rounded-xl font-semibold transition-colors">
                  Start Guided Check
                </button>
              </div>
            )}

            {soundState === "positioning" && (
              <div className="flex flex-col items-center gap-4">
                <ZoneMap activeZone={ZONE_ORDER[zoneIndex]} />
                <div className="text-center">
                  <p className="text-slate-300 text-sm font-medium">{ZONE_INFO[ZONE_ORDER[zoneIndex]].title}</p>
                  <p className="text-slate-500 text-xs mt-1">{ZONE_INFO[ZONE_ORDER[zoneIndex]].instruction}</p>
                  {zoneMessage && <p className="text-amber-400/80 text-xs mt-2">{zoneMessage}</p>}
                </div>
                <button onClick={beginZoneCheck} className="w-full py-3 bg-blue-500 hover:bg-blue-600 rounded-xl font-semibold transition-colors">
                  I'm in position
                </button>
                <button onClick={stopGuidedFlow} className="text-slate-500 text-xs underline">
                  Cancel
                </button>
              </div>
            )}

            {soundState === "checking" && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-400 font-medium text-sm animate-pulse">Checking {ZONE_INFO[ZONE_ORDER[zoneIndex]].title.toLowerCase()}</p>
                    <p className="text-slate-500 text-xs">Hold still, keep the room quiet</p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-lg font-bold">{checkCountdown}</span>
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
                  <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${((CHECK_DURATION - checkCountdown) / CHECK_DURATION) * 100}%` }} />
                </div>

                <button onClick={stopGuidedFlow} className="text-slate-500 text-xs underline self-center">
                  Cancel
                </button>
              </div>
            )}

            {soundState === "recording" && (
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-400 font-medium text-sm animate-pulse">{zoneMessage ?? "Great signal! Recording a bit more..."}</p>
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
                  <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${((EXTEND_DURATION - soundCountdown) / EXTEND_DURATION) * 100}%` }} />
                </div>

                <button onClick={stopGuidedFlow} className="text-slate-500 text-xs underline self-center">
                  Cancel
                </button>
              </div>
            )}

            {soundState === "all_zones_failed" && (
              <div className="flex flex-col items-center gap-4">
                <div className="w-14 h-14 rounded-full bg-amber-500/10 border-2 border-amber-500/40 flex items-center justify-center">
                  <svg className="w-7 h-7 text-amber-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" /></svg>
                </div>
                <div className="text-center">
                  <p className="text-slate-300 text-sm font-medium">Couldn't get a clean signal at any spot</p>
                  <p className="text-slate-500 text-xs mt-1">You can use the best of the 3 recordings anyway (result may be less reliable), or try the guided check again.</p>
                </div>
                <button onClick={useBestZoneAnyway} className="w-full py-3 bg-blue-500 hover:bg-blue-600 rounded-xl font-semibold transition-colors">
                  Use best spot anyway
                </button>
                <button onClick={retryGuidedFlow} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Try again
                </button>
                <button onClick={stopGuidedFlow} className="text-slate-500 text-xs underline">
                  Cancel
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
                <button onClick={() => { setSoundState("idle"); setSoundResult(null); setSoundWaveform([]); setZoneIndex(0); zoneChunksRef.current = {}; zoneScoresRef.current = {}; }} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
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
            {!soundResult ? (
              <div className="text-center py-6">
                <p className="text-slate-500 text-sm">Complete the heart sound check to generate a report</p>
                <div className="flex gap-2 mt-3 justify-center">
                  <button onClick={() => setTab("sound")} className="py-2 px-4 bg-blue-500/20 text-blue-400 rounded-lg text-xs">Heart Sound</button>
                </div>
              </div>
            ) : (
              <>
                <div className="bg-slate-700/30 rounded-xl p-4 text-center">
                  <div className="text-xs text-slate-500 mb-1">Overall Screening</div>
                  {(() => {
                    const score = soundResult.label === "abnormal" ? 0 : 100;
                    const color = score >= 80 ? "text-green-400" : "text-red-400";
                    const label = score >= 80 ? "No obvious flags" : "Worth a check-up";
                    return (<><div className={`text-4xl font-bold ${color}`}>{score}%</div><div className={`text-sm font-medium ${color}`}>{label}</div></>);
                  })()}
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                    <span className="text-xs text-slate-400">Heart Sound</span>
                    <span className={`text-sm font-bold ${soundResult.label === "normal" ? "text-green-400" : "text-yellow-400"}`}>{soundResult.label === "normal" ? "Normal" : "Atypical"} ({(soundResult.confidence * 100).toFixed(0)}%)</span>
                  </div>
                </div>
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
