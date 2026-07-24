"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// ============ CONSTANTS ============
const SAMPLE_RATE = 2000;
const SEGMENT_DURATION = 3;
const RECORD_DURATION = 20;
const N_MFCC = 20;
const HOP_LENGTH = 128;
const MAX_FRAMES = 47;
const N_FFT = 512;
const N_MELS = 40;
const PPG_DURATION = 60;
const PPG_FPS = 30;
const WAVEFORM_POINTS = 300; // points visible in live waveform

type Tab = "pulse" | "sound" | "report";
type PulseState = "idle" | "measuring" | "processing" | "done";
type SoundState = "idle" | "recording" | "processing" | "done";

interface PulseResult {
  hr: number;
  hrv: number;
  rrIntervals: number[];
  irregularity: number;
  rhythm: "regular" | "irregular" | "highly_irregular";
  signal: number[];
}

// Heart sound DSP analysis findings
interface HeartSoundFindings {
  heartRate: number;
  s1s2Detected: boolean;
  s3Suspected: boolean;
  s4Suspected: boolean;
  systolicMurmur: boolean;
  diastolicMurmur: boolean;
  extraSounds: boolean;
  findings: string[];
  s1s2Ratio: number;
  systolicEnergy: number;
  diastolicEnergy: number;
}

interface QualityReport {
  pass: boolean;
  score: number; // 0-100
  issues: string[];
  suggestions: string[];
}

interface SoundResult {
  label: string;
  confidence: number;
  normal: number;
  abnormal: number;
  segmentsAnalyzed: number;
  bestSegmentIdx: number;
  // DSP analysis
  heartSoundFindings: HeartSoundFindings | null;
  qualityReport: QualityReport | null;
}

// ============ DSP UTILITIES ============

function hann(N: number): Float32Array {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

function rfft(frame: Float32Array): { re: Float32Array; im: Float32Array } {
  const N = frame.length;
  const re = new Float32Array(N / 2 + 1);
  const im = new Float32Array(N / 2 + 1);
  for (let k = 0; k <= N / 2; k++) {
    let sumRe = 0, sumIm = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      sumRe += frame[n] * Math.cos(angle);
      sumIm -= frame[n] * Math.sin(angle);
    }
    re[k] = sumRe;
    im[k] = sumIm;
  }
  return { re, im };
}

function powerSpectrum(frame: Float32Array): Float32Array {
  const { re, im } = rfft(frame);
  const ps = new Float32Array(re.length);
  for (let i = 0; i < re.length; i++) ps[i] = (re[i] * re[i] + im[i] * im[i]) / frame.length;
  return ps;
}

function melFilterbank(sr: number, nFft: number, nMels: number): Float32Array[] {
  const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
  const melToHz = (mel: number) => 700 * (Math.pow(10, mel / 2595) - 1);
  const melLow = hzToMel(0);
  const melHigh = hzToMel(sr / 2);
  const melPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) melPoints[i] = melLow + ((melHigh - melLow) * i) / (nMels + 1);
  const binPoints = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) binPoints[i] = Math.floor(((nFft + 1) * melToHz(melPoints[i])) / sr);
  const filters: Float32Array[] = [];
  const nBins = nFft / 2 + 1;
  for (let m = 0; m < nMels; m++) {
    const fb = new Float32Array(nBins);
    const start = binPoints[m], center = binPoints[m + 1], end = binPoints[m + 2];
    for (let k = Math.floor(start); k < Math.ceil(center); k++) if (k < nBins && center !== start) fb[k] = (k - start) / (center - start);
    for (let k = Math.floor(center); k < Math.ceil(end); k++) if (k < nBins && end !== center) fb[k] = (end - k) / (end - center);
    filters.push(fb);
  }
  return filters;
}

function dctMatrix(nMfcc: number, nMels: number): Float32Array[] {
  const matrix: Float32Array[] = [];
  for (let i = 0; i < nMfcc; i++) {
    const row = new Float32Array(nMels);
    for (let j = 0; j < nMels; j++) row[j] = Math.cos((Math.PI * i * (j + 0.5)) / nMels);
    matrix.push(row);
  }
  return matrix;
}

function extractMfcc(audio: Float32Array): Float32Array[] {
  const window = hann(N_FFT);
  const filters = melFilterbank(SAMPLE_RATE, N_FFT, N_MELS);
  const dct = dctMatrix(N_MFCC, N_MELS);
  const frames: Float32Array[] = [];
  for (let start = 0; start + N_FFT <= audio.length; start += HOP_LENGTH) {
    const frame = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) frame[i] = audio[start + i] * window[i];
    const ps = powerSpectrum(frame);
    const melEnergies = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let sum = 0;
      for (let k = 0; k < ps.length; k++) sum += ps[k] * filters[m][k];
      melEnergies[m] = Math.log(Math.max(sum, 1e-10));
    }
    const mfcc = new Float32Array(N_MFCC);
    for (let i = 0; i < N_MFCC; i++) {
      let sum = 0;
      for (let j = 0; j < N_MELS; j++) sum += dct[i][j] * melEnergies[j];
      mfcc[i] = sum;
    }
    frames.push(mfcc);
  }
  return frames;
}

// ============ Heart Sound DSP Analysis ============

// Simple 2nd-order IIR bandpass filter (biquad)
function biquadBandpass(audio: Float32Array, sr: number, centerFreq: number, Q: number): Float32Array {
  const w0 = 2 * Math.PI * centerFreq / sr;
  const alpha = Math.sin(w0) / (2 * Q);
  const b0 = alpha, b1 = 0, b2 = -alpha;
  const a0 = 1 + alpha, a1 = -2 * Math.cos(w0), a2 = 1 - alpha;
  const out = new Float32Array(audio.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < audio.length; i++) {
    const x0 = audio[i];
    out[i] = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x0; y2 = y1; y1 = out[i];
  }
  return out;
}

// Compute amplitude envelope: rectify + lowpass smooth
function amplitudeEnvelope(audio: Float32Array, smoothSamples: number): Float32Array {
  const rectified = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) rectified[i] = Math.abs(audio[i]);
  // Moving average lowpass
  const env = new Float32Array(audio.length);
  let sum = 0;
  for (let i = 0; i < audio.length; i++) {
    sum += rectified[i];
    if (i >= smoothSamples) sum -= rectified[i - smoothSamples];
    env[i] = sum / Math.min(i + 1, smoothSamples);
  }
  return env;
}

// Autocorrelation-based heart rate detection on envelope
function autocorrelationHR(envelope: Float32Array, sr: number): { hr: number; quality: number } {
  // Search range: 40-180 BPM → period 0.33s-1.5s
  const minLag = Math.round(sr * 0.33);  // 180 BPM
  const maxLag = Math.round(sr * 1.5);   // 40 BPM

  // Remove mean
  let mean = 0;
  for (let i = 0; i < envelope.length; i++) mean += envelope[i];
  mean /= envelope.length;
  const centered = new Float32Array(envelope.length);
  for (let i = 0; i < envelope.length; i++) centered[i] = envelope[i] - mean;

  // Compute autocorrelation for lags in heart rate range
  let bestLag = minLag;
  let bestCorr = -1;
  let zeroLagCorr = 0;

  // Zero-lag (normalization)
  for (let i = 0; i < centered.length; i++) zeroLagCorr += centered[i] * centered[i];

  for (let lag = minLag; lag <= Math.min(maxLag, centered.length - 1); lag++) {
    let corr = 0;
    const n = centered.length - lag;
    for (let i = 0; i < n; i++) corr += centered[i] * centered[i + lag];
    const normalized = zeroLagCorr > 0 ? corr / zeroLagCorr : 0;
    if (normalized > bestCorr) {
      bestCorr = normalized;
      bestLag = lag;
    }
  }

  const hr = bestLag > 0 ? Math.round(60 * sr / bestLag) : 0;
  // Quality: autocorrelation peak strength (0-1). Above 0.3 = decent periodicity
  const quality = Math.max(0, bestCorr);
  return { hr, quality };
}

// Find peaks in envelope with adaptive threshold
function findEnvelopePeaks(envelope: Float32Array, minDist: number): number[] {
  // Adaptive threshold: median + 0.5 * (max - median) of the envelope
  const sorted = Array.from(envelope).sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  const max = sorted[sorted.length - 1];
  const threshold = median + 0.5 * (max - median);

  const peaks: number[] = [];
  for (let i = 2; i < envelope.length - 2; i++) {
    if (envelope[i] > threshold &&
        envelope[i] >= envelope[i-1] && envelope[i] >= envelope[i+1] &&
        envelope[i] >= envelope[i-2] && envelope[i] >= envelope[i+2]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

// ============ Noise Reduction ============

// Spectral subtraction: estimate noise from quietest segments, subtract from signal
function spectralSubtraction(audio: Float32Array, sr: number, fftSize: number = 512): Float32Array {
  const hopSize = fftSize / 2;
  const numFrames = Math.floor((audio.length - fftSize) / hopSize) + 1;
  if (numFrames < 4) return audio;

  const win = hann(fftSize);

  // Helper: compute magnitude spectrum of a frame
  function frameMagnitude(start: number): Float32Array {
    const { re, im } = rfft((() => {
      const f = new Float32Array(fftSize);
      for (let i = 0; i < fftSize; i++) f[i] = (start + i < audio.length ? audio[start + i] : 0) * win[i];
      return f;
    })());
    const mag = new Float32Array(re.length);
    for (let i = 0; i < re.length; i++) mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    return mag;
  }

  // Estimate noise profile from the 20% quietest frames (by RMS energy)
  const frameEnergies: { start: number; rms: number }[] = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    let rms = 0;
    for (let i = 0; i < fftSize && start + i < audio.length; i++) rms += audio[start + i] * audio[start + i];
    frameEnergies.push({ start, rms: Math.sqrt(rms / fftSize) });
  }
  frameEnergies.sort((a, b) => a.rms - b.rms);
  const noiseFrameCount = Math.max(2, Math.floor(numFrames * 0.2));

  // Average noise magnitude spectrum
  const noiseMag = new Float32Array(fftSize / 2 + 1);
  for (let i = 0; i < noiseFrameCount; i++) {
    const mag = frameMagnitude(frameEnergies[i].start);
    for (let k = 0; k < noiseMag.length; k++) noiseMag[k] += mag[k];
  }
  for (let k = 0; k < noiseMag.length; k++) noiseMag[k] /= noiseFrameCount;

  // Subtract noise from each frame using overlap-add
  const output = new Float32Array(audio.length);
  const windowSum = new Float32Array(audio.length);
  const overSubFactor = 2.0; // over-subtraction factor for aggressive noise removal

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSize;
    const frame = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) frame[i] = (start + i < audio.length ? audio[start + i] : 0) * win[i];

    const { re, im } = rfft(frame);
    const mag = new Float32Array(re.length);
    const phase = new Float32Array(re.length);
    for (let k = 0; k < re.length; k++) {
      mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      phase[k] = Math.atan2(im[k], re[k]);
    }

    // Subtract noise magnitude (spectral floor at 0.05 * original to avoid musical noise)
    const cleanMag = new Float32Array(re.length);
    for (let k = 0; k < re.length; k++) {
      cleanMag[k] = Math.max(mag[k] - overSubFactor * noiseMag[k], 0.05 * mag[k]);
    }

    // Reconstruct frame via inverse FFT (simplified: use magnitude + original phase)
    const cleanFrame = new Float32Array(fftSize);
    // Inverse DFT
    for (let n = 0; n < fftSize; n++) {
      let val = 0;
      for (let k = 0; k < re.length; k++) {
        const angle = (2 * Math.PI * k * n) / fftSize;
        val += cleanMag[k] * Math.cos(angle + phase[k]);
        if (k > 0 && k < fftSize / 2) val += cleanMag[k] * Math.cos(angle - phase[k]); // mirror
      }
      cleanFrame[n] = (val / fftSize) * win[n];
    }

    // Overlap-add
    for (let i = 0; i < fftSize && start + i < audio.length; i++) {
      output[start + i] += cleanFrame[i];
      windowSum[start + i] += win[i] * win[i];
    }
  }

  // Normalize by window sum
  for (let i = 0; i < audio.length; i++) {
    output[i] = windowSum[i] > 0.01 ? output[i] / windowSum[i] : 0;
  }

  return output;
}

// LMS Adaptive Line Enhancer: extracts periodic components (heartbeats) from noise
function lmsAdaptiveLineEnhancer(audio: Float32Array, filterLength: number = 32, delay: number = 15, mu: number = 0.005): Float32Array {
  const output = new Float32Array(audio.length);
  const weights = new Float32Array(filterLength);
  const buffer = new Float32Array(filterLength + delay);

  for (let n = 0; n < audio.length; n++) {
    // Shift buffer
    buffer.copyWithin(1, 0, buffer.length - 1);
    buffer[0] = audio[n];

    // Reference: delayed version of input (decorrelated noise, correlated heartbeat)
    let y = 0;
    for (let k = 0; k < filterLength; k++) {
      const refIdx = k + delay;
      if (refIdx < buffer.length) y += weights[k] * buffer[refIdx];
    }

    // Error
    const e = audio[n] - y;

    // Normalize step size by signal power to prevent divergence
    let power = 0;
    for (let k = 0; k < filterLength; k++) {
      const refIdx = k + delay;
      if (refIdx < buffer.length) power += buffer[refIdx] * buffer[refIdx];
    }
    const normMu = power > 1e-10 ? mu / (power + 1e-6) : mu;

    // LMS weight update
    for (let k = 0; k < filterLength; k++) {
      const refIdx = k + delay;
      if (refIdx < buffer.length) weights[k] += 2 * normMu * e * buffer[refIdx];
    }

    output[n] = y; // Output = predicted periodic component (heart sounds)
  }

  return output;
}

// 4th order Butterworth bandpass via cascaded biquads
function butterworthBandpass(audio: Float32Array, sr: number, lowFreq: number, highFreq: number): Float32Array {
  // Stage 1: highpass
  let result = biquadFilter(audio, sr, 'highpass', lowFreq, 0.707);
  result = biquadFilter(result, sr, 'highpass', lowFreq, 0.707);
  // Stage 2: lowpass
  result = biquadFilter(result, sr, 'lowpass', highFreq, 0.707);
  result = biquadFilter(result, sr, 'lowpass', highFreq, 0.707);
  return result;
}

function biquadFilter(audio: Float32Array, sr: number, type: 'highpass' | 'lowpass', freq: number, Q: number): Float32Array {
  const w0 = 2 * Math.PI * freq / sr;
  const alpha = Math.sin(w0) / (2 * Q);
  let b0: number, b1: number, b2: number, a0: number, a1: number, a2: number;

  if (type === 'highpass') {
    b0 = (1 + Math.cos(w0)) / 2;
    b1 = -(1 + Math.cos(w0));
    b2 = (1 + Math.cos(w0)) / 2;
  } else {
    b0 = (1 - Math.cos(w0)) / 2;
    b1 = 1 - Math.cos(w0);
    b2 = (1 - Math.cos(w0)) / 2;
  }
  a0 = 1 + alpha;
  a1 = -2 * Math.cos(w0);
  a2 = 1 - alpha;

  const out = new Float32Array(audio.length);
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < audio.length; i++) {
    const x0 = audio[i];
    out[i] = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
    x2 = x1; x1 = x0; y2 = y1; y1 = out[i];
  }
  return out;
}

// Shannon energy envelope with proper lowpass at ~8Hz
function shannonEnergyEnvelope(audio: Float32Array, sr: number): Float32Array {
  // Compute Shannon energy per sample
  const energy = new Float32Array(audio.length);
  for (let i = 0; i < audio.length; i++) {
    const x2 = audio[i] * audio[i];
    energy[i] = x2 < 1e-12 ? 0 : -x2 * Math.log(x2 + 1e-12);
  }
  // Lowpass at ~8Hz via moving average (window = sr/8)
  const windowSize = Math.round(sr / 8);
  const env = new Float32Array(audio.length);
  let sum = 0;
  for (let i = 0; i < audio.length; i++) {
    sum += energy[i];
    if (i >= windowSize) sum -= energy[i - windowSize];
    env[i] = sum / Math.min(i + 1, windowSize);
  }
  // Normalize
  let maxVal = 0;
  for (let i = 0; i < env.length; i++) if (env[i] > maxVal) maxVal = env[i];
  if (maxVal > 0) for (let i = 0; i < env.length; i++) env[i] /= maxVal;
  return env;
}

// Quality assessment gate — checks signal before analysis
function assessSignalQuality(audio: Float32Array, sr: number): QualityReport {
  const issues: string[] = [];
  const suggestions: string[] = [];
  let score = 100;

  // QA1: Signal presence — RMS energy above threshold
  let rms = 0;
  for (let i = 0; i < audio.length; i++) rms += audio[i] * audio[i];
  rms = Math.sqrt(rms / audio.length);
  if (rms < 0.001) {
    issues.push("No signal detected");
    suggestions.push("Press phone mic firmly against bare chest");
    score -= 40;
  } else if (rms < 0.005) {
    issues.push("Very weak signal");
    suggestions.push("Press mic more firmly, remove phone case");
    score -= 20;
  }

  // QA2: Clipping detection
  let clipCount = 0;
  for (let i = 0; i < audio.length; i++) {
    if (Math.abs(audio[i]) > 0.98) clipCount++;
  }
  if (clipCount > audio.length * 0.01) {
    issues.push("Audio clipping detected");
    suggestions.push("Reduce pressure slightly or move to quieter environment");
    score -= 15;
  }

  // QA3: Frequency content — energy ratio in 20-200Hz vs total
  const heartBand = butterworthBandpass(audio, sr, 20, 200);
  let heartEnergy = 0, totalEnergy = 0;
  for (let i = 0; i < audio.length; i++) {
    heartEnergy += heartBand[i] * heartBand[i];
    totalEnergy += audio[i] * audio[i];
  }
  const heartRatio = totalEnergy > 0 ? heartEnergy / totalEnergy : 0;
  if (heartRatio < 0.15) {
    issues.push("Low cardiac frequency content");
    suggestions.push("Mic may not be on chest — position between ribs, left of sternum");
    score -= 25;
  }

  // QA4: Periodicity check via autocorrelation
  const envelope = shannonEnergyEnvelope(audio, sr);
  const { quality: acQuality } = autocorrelationHR(envelope, sr);
  if (acQuality < 0.08) {
    issues.push("No periodic heartbeat pattern found");
    suggestions.push("Ensure mic is pressed firmly on bare chest, stay still");
    score -= 30;
  } else if (acQuality < 0.15) {
    issues.push("Weak periodicity");
    suggestions.push("Hold still, reduce background noise");
    score -= 15;
  }

  // QA5: Motion artifact — check short-term energy variance
  const chunkSize = Math.floor(sr * 0.5); // 0.5s chunks
  const numChunks = Math.floor(audio.length / chunkSize);
  if (numChunks >= 4) {
    const chunkRMS: number[] = [];
    for (let c = 0; c < numChunks; c++) {
      let e = 0;
      for (let i = c * chunkSize; i < (c + 1) * chunkSize; i++) e += audio[i] * audio[i];
      chunkRMS.push(Math.sqrt(e / chunkSize));
    }
    const meanRMS = chunkRMS.reduce((a, b) => a + b) / chunkRMS.length;
    const rmsVariance = chunkRMS.reduce((s, r) => s + (r - meanRMS) ** 2, 0) / chunkRMS.length;
    const rmsCV = meanRMS > 0 ? Math.sqrt(rmsVariance) / meanRMS : 0;
    if (rmsCV > 0.8) {
      issues.push("Motion artifact detected");
      suggestions.push("Keep phone and body completely still during recording");
      score -= 20;
    }
  }

  score = Math.max(0, Math.min(100, score));
  return { pass: score >= 60, score, issues, suggestions };
}

// Cascaded Sign Error LMS — 3 stages with variable step sizes
// Based on research: provides 10-17dB SNR improvement over single-stage
function cascadedSELMS(audio: Float32Array, filterLength: number = 32, delay: number = 15, baseMu: number = 0.01): Float32Array {
  const numStages = 3;
  const stepFactor = 2; // step size doubles each stage

  let currentInput = audio;
  let output = new Float32Array(audio.length);

  for (let stage = 0; stage < numStages; stage++) {
    const mu = baseMu * Math.pow(stepFactor, stage);
    const weights = new Float32Array(filterLength);
    const buffer = new Float32Array(filterLength + delay);
    const stageOutput = new Float32Array(audio.length);
    const error = new Float32Array(audio.length);

    for (let n = 0; n < currentInput.length; n++) {
      // Shift buffer
      buffer.copyWithin(1, 0, buffer.length - 1);
      buffer[0] = currentInput[n];

      // Compute filter output from delayed reference
      let y = 0;
      let power = 0;
      for (let k = 0; k < filterLength; k++) {
        const refIdx = k + delay;
        if (refIdx < buffer.length) {
          y += weights[k] * buffer[refIdx];
          power += buffer[refIdx] * buffer[refIdx];
        }
      }

      stageOutput[n] = y;
      error[n] = currentInput[n] - y;

      // Sign Error LMS update (computationally cheaper than standard LMS)
      const signErr = error[n] > 0 ? 1 : error[n] < 0 ? -1 : 0;
      const normMu = power > 1e-10 ? mu / (power + 1e-6) : mu;
      for (let k = 0; k < filterLength; k++) {
        const refIdx = k + delay;
        if (refIdx < buffer.length) {
          weights[k] += normMu * signErr * buffer[refIdx];
        }
      }
    }

    output = stageOutput;
    currentInput = error; // Next stage processes the residual
  }

  return output;
}

// Full noise reduction pipeline: bandpass → spectral subtraction → Cascaded SE-LMS
function cleanHeartSound(audio: Float32Array, sr: number): { cleaned: Float32Array; snrImprovement: number } {
  // Step 1: 4th order Butterworth bandpass 20-200Hz (S1/S2 heart-sound band).
  // Narrower than the old 25-400Hz: cutting 200-400Hz removes a lot of room/rustle
  // noise while keeping the fundamental heart sounds, which improves SNR on phone mics.
  const bandpassed = butterworthBandpass(audio, sr, 20, 200);

  // Step 2: Spectral subtraction to remove stationary noise
  const specSub = spectralSubtraction(bandpassed, sr, 256);

  // Step 3: Cascaded SE-LMS to extract periodic heartbeat (3-stage, 10-17dB SNR gain)
  const cleaned = cascadedSELMS(specSub, 48, 15, 0.008);

  // Compute SNR improvement estimate
  let signalPower = 0, noisePower = 0;
  for (let i = 0; i < audio.length; i++) {
    signalPower += cleaned[i] * cleaned[i];
    noisePower += (bandpassed[i] - cleaned[i]) * (bandpassed[i] - cleaned[i]);
  }
  const snrImprovement = noisePower > 0 ? 10 * Math.log10(signalPower / noisePower) : 0;

  return { cleaned, snrImprovement };
}

// Analyze heart sounds for S3, S4, murmurs
function analyzeHeartSounds(audio: Float32Array, sr: number): HeartSoundFindings {
  const findings: string[] = [];
  const result: HeartSoundFindings = {
    heartRate: 0, s1s2Detected: false, s3Suspected: false, s4Suspected: false,
    systolicMurmur: false, diastolicMurmur: false, extraSounds: false,
    findings, s1s2Ratio: 1, systolicEnergy: 0, diastolicEnergy: 0
  };

  // ---- STEP 1: Full noise reduction pipeline ----
  const { cleaned, snrImprovement } = cleanHeartSound(audio, sr);
  findings.push(`Signal processed: bandpass 25-400Hz → spectral subtraction → cascaded SE-LMS (SNR gain: ${snrImprovement.toFixed(1)}dB)`);

  // ---- STEP 2: Shannon energy envelope (research-backed, best for S1/S2 detection) ----
  const envelope = shannonEnergyEnvelope(cleaned, sr);

  if (envelope.length < sr * 2) {
    findings.push("Recording too short for reliable analysis");
    return result;
  }

  // ---- STEP 3: Autocorrelation-based heart rate on cleaned signal ----
  const { hr: acHR, quality: acQuality } = autocorrelationHR(envelope, sr);
  result.heartRate = acHR;

  // Quality gate
  if (acQuality < 0.10) {
    findings.push("Could not detect heart sounds even after noise reduction. Tips: press phone mic firmly on bare chest (left side, between ribs), hold breath for 10 seconds, stay completely still, quiet room.");
    return result;
  }

  if (acQuality < 0.20) {
    findings.push("Weak heart sound signal — results less reliable. Press mic more firmly, hold breath.");
  }

  result.s1s2Detected = true;
  findings.push(`Heart rate from auscultation: ~${acHR} BPM (signal quality: ${(acQuality * 100).toFixed(0)}%)`);

  // ---- STEP 4: Peak detection using autocorrelation-derived period ----
  const expectedPeriod = sr * 60 / acHR; // expected samples per heartbeat
  // Min distance between S1/S2 peaks: ~40% of beat period (S1-S2 systole is ~35% of cycle)
  const minPeakDist = Math.round(expectedPeriod * 0.25);
  const peaks = findEnvelopePeaks(envelope, minPeakDist);

  if (peaks.length < 6) {
    findings.push("Too few heart sound peaks detected for detailed analysis");
    return result;
  }

  // ---- STEP 5: Identify S1 and S2 by interval pattern ----
  const peakTimes = peaks.map(p => p / sr);
  const intervals: number[] = [];
  for (let i = 1; i < peakTimes.length; i++) intervals.push(peakTimes[i] - peakTimes[i-1]);

  if (intervals.length < 4) return result;

  const medianInterval = [...intervals].sort((a,b) => a-b)[Math.floor(intervals.length/2)];

  const s1Indices: number[] = [];
  const s2Indices: number[] = [];
  for (let i = 0; i < peaks.length - 1; i++) {
    const interval = peakTimes[i+1] - peakTimes[i];
    if (interval < medianInterval) {
      s1Indices.push(i);
      s2Indices.push(i + 1);
    }
  }

  if (s1Indices.length < 3 || s2Indices.length < 3) {
    findings.push("Could not reliably separate S1/S2 — try recording in a quieter environment");
    return result;
  }

  // S1/S2 amplitude ratio
  let s1Amp = 0, s2Amp = 0;
  for (const idx of s1Indices) s1Amp += envelope[peaks[idx]];
  for (const idx of s2Indices) s2Amp += envelope[peaks[idx]];
  s1Amp /= s1Indices.length;
  s2Amp /= s2Indices.length;
  result.s1s2Ratio = s2Amp > 0 ? s1Amp / s2Amp : 1;

  // ---- STEP 6: Compute quiet baseline from mid-diastole ----
  const quietEnergies: number[] = [];
  for (let i = 0; i < s2Indices.length - 1; i++) {
    const s2Pos = peaks[s2Indices[i]];
    const nextS1Pos = s1Indices[i + 1] !== undefined ? peaks[s1Indices[i + 1]] : null;
    if (nextS1Pos !== null) {
      const diaLen = nextS1Pos - s2Pos;
      // Middle 30% of diastole — should be the quietest
      const midStart = s2Pos + Math.floor(diaLen * 0.4);
      const midEnd = s2Pos + Math.floor(diaLen * 0.7);
      if (midEnd <= envelope.length && midEnd > midStart) {
        let sum = 0;
        for (let k = midStart; k < midEnd; k++) sum += envelope[k];
        quietEnergies.push(sum / (midEnd - midStart));
      }
    }
  }

  if (quietEnergies.length === 0) return result;
  const quietBaseline = quietEnergies.reduce((a, b) => a + b) / quietEnergies.length;
  const avgPeakAmp = (s1Amp + s2Amp) / 2;

  // Signal-to-noise ratio: how much louder are heart sounds vs quiet periods
  const snr = quietBaseline > 0 ? avgPeakAmp / quietBaseline : 0;
  if (snr < 3) {
    findings.push("Low signal-to-noise ratio — heart sounds barely above background noise. Press mic firmly on bare skin.");
    return result;
  }

  // ---- STEP 7: S3 detection (early diastole, 120-200ms after S2) ----
  let s3Count = 0, s3Total = 0;
  for (const s2Idx of s2Indices) {
    const s2Pos = peaks[s2Idx];
    const s3Start = s2Pos + Math.round(0.12 * sr);
    const s3End = s2Pos + Math.round(0.20 * sr);
    if (s3End >= envelope.length) continue;
    s3Total++;

    let maxInWindow = 0;
    for (let k = s3Start; k < s3End; k++) {
      if (envelope[k] > maxInWindow) maxInWindow = envelope[k];
    }
    // S3 must be: (a) ≥40% of S2 peak, (b) ≥6x quiet baseline, (c) have a clear bump shape
    const s2PeakAmp = envelope[peaks[s2Idx]];
    if (maxInWindow > s2PeakAmp * 0.40 && maxInWindow > quietBaseline * 6) {
      s3Count++;
    }
  }
  // Must appear consistently (75%+ of cycles) with enough cycles to judge
  if (s3Total >= 5 && s3Count >= Math.ceil(s3Total * 0.75)) {
    result.s3Suspected = true;
    result.extraSounds = true;
    findings.push("S3 gallop suspected — may indicate heart failure or volume overload (common post-MI)");
  }

  // ---- STEP 8: S4 detection (late diastole, 60-130ms before S1) ----
  let s4Count = 0, s4Total = 0;
  for (const s1Idx of s1Indices) {
    const s1Pos = peaks[s1Idx];
    const s4Start = s1Pos - Math.round(0.13 * sr);
    const s4End = s1Pos - Math.round(0.06 * sr);
    if (s4Start < 0) continue;
    s4Total++;

    let maxInWindow = 0;
    for (let k = s4Start; k < s4End; k++) {
      if (envelope[k] > maxInWindow) maxInWindow = envelope[k];
    }
    const s1PeakAmp = envelope[peaks[s1Idx]];
    if (maxInWindow > s1PeakAmp * 0.35 && maxInWindow > quietBaseline * 6) {
      s4Count++;
    }
  }
  if (s4Total >= 5 && s4Count >= Math.ceil(s4Total * 0.75)) {
    result.s4Suspected = true;
    result.extraSounds = true;
    findings.push("S4 gallop suspected — may indicate stiff ventricle (common in acute MI, hypertension)");
  }

  if (result.s3Suspected && result.s4Suspected) {
    findings.push("Summation gallop pattern — strongly suggests cardiac dysfunction, urgent evaluation needed");
  }

  // ---- STEP 9: Murmur detection ----
  let systolicEnergyTotal = 0, diastolicEnergyTotal = 0;
  let systolicCount = 0, diastolicCount = 0;
  const margin = Math.round(0.03 * sr); // 30ms margin from peaks

  for (let i = 0; i < s1Indices.length && i < s2Indices.length; i++) {
    const s1Pos = peaks[s1Indices[i]];
    const s2Pos = peaks[s2Indices[i]];
    const sysStart = s1Pos + margin;
    const sysEnd = s2Pos - margin;
    if (sysEnd > sysStart) {
      let sum = 0;
      for (let k = sysStart; k < sysEnd; k++) sum += envelope[k];
      systolicEnergyTotal += sum / (sysEnd - sysStart);
      systolicCount++;
    }
  }

  for (let i = 0; i < s2Indices.length - 1; i++) {
    const s2Pos = peaks[s2Indices[i]];
    const nextS1Pos = s1Indices[i + 1] !== undefined ? peaks[s1Indices[i + 1]] : null;
    if (nextS1Pos !== null) {
      const diaStart = s2Pos + margin;
      const diaEnd = nextS1Pos - margin;
      if (diaEnd > diaStart) {
        let sum = 0;
        for (let k = diaStart; k < diaEnd; k++) sum += envelope[k];
        diastolicEnergyTotal += sum / (diaEnd - diaStart);
        diastolicCount++;
      }
    }
  }

  const avgSysEnergy = systolicCount > 0 ? systolicEnergyTotal / systolicCount : 0;
  const avgDiaEnergy = diastolicCount > 0 ? diastolicEnergyTotal / diastolicCount : 0;
  result.systolicEnergy = avgSysEnergy;
  result.diastolicEnergy = avgDiaEnergy;

  // Murmur = sustained energy between sounds significantly above quiet baseline
  const murmurThreshold = quietBaseline * 6;
  if (avgSysEnergy > murmurThreshold && systolicCount >= 3) {
    result.systolicMurmur = true;
    findings.push("Systolic murmur pattern detected — may indicate valve disease or post-MI complication");
  }
  if (avgDiaEnergy > murmurThreshold && diastolicCount >= 3) {
    result.diastolicMurmur = true;
    findings.push("Diastolic murmur pattern detected — may indicate valve insufficiency");
  }

  // ---- STEP 10: Additional findings ----
  if (acHR > 100) findings.push("Tachycardia detected — elevated rate may indicate stress or cardiac compensation");
  else if (acHR < 50) findings.push("Bradycardia detected from heart sounds");

  if (result.s1s2Ratio > 3.0) {
    findings.push("S1 significantly louder than S2 — may indicate mitral stenosis");
  } else if (result.s1s2Ratio < 0.3) {
    findings.push("S2 significantly louder than S1 — may indicate hypertension or aortic disease");
  }

  if (!result.s3Suspected && !result.s4Suspected && !result.systolicMurmur && !result.diastolicMurmur) {
    findings.push("No additional abnormal sounds detected in this recording");
  }

  return result;
}

// ============ PPG Signal Processing ============

function smoothSignal(signal: number[], windowSize: number): number[] {
  const result = new Array(signal.length).fill(0);
  for (let i = 0; i < signal.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(signal.length, i + Math.floor(windowSize / 2) + 1);
    let sum = 0;
    for (let j = start; j < end; j++) sum += signal[j];
    result[i] = sum / (end - start);
  }
  return result;
}

function detrendSignal(signal: number[], fps: number): number[] {
  // Remove slow drift with a 2-second moving average subtraction
  const trendWindow = Math.round(fps * 2);
  const trend = smoothSignal(signal, trendWindow);
  return signal.map((v, i) => v - trend[i]);
}

// FFT-based heart rate detection (much more robust than peak detection)
function fftHeartRate(signal: number[], fps: number): number {
  const N = signal.length;
  // Zero-pad to next power of 2 for cleaner FFT
  let fftSize = 1;
  while (fftSize < N) fftSize *= 2;

  // Apply Hann window and zero-pad
  const windowed = new Array(fftSize).fill(0);
  for (let i = 0; i < N; i++) {
    windowed[i] = signal[i] * (0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1))));
  }

  // Compute magnitude spectrum (DFT - only positive freqs)
  const halfN = Math.floor(fftSize / 2);
  const magnitudes = new Array(halfN).fill(0);
  for (let k = 0; k < halfN; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < fftSize; n++) {
      const angle = (2 * Math.PI * k * n) / fftSize;
      re += windowed[n] * Math.cos(angle);
      im -= windowed[n] * Math.sin(angle);
    }
    magnitudes[k] = Math.sqrt(re * re + im * im);
  }

  // Find peak in 0.8-3.0 Hz range (48-180 BPM)
  const minBin = Math.floor(0.8 * fftSize / fps);
  const maxBin = Math.ceil(3.0 * fftSize / fps);

  let peakBin = minBin;
  let peakMag = 0;
  for (let k = minBin; k <= Math.min(maxBin, halfN - 1); k++) {
    if (magnitudes[k] > peakMag) {
      peakMag = magnitudes[k];
      peakBin = k;
    }
  }

  // Parabolic interpolation for sub-bin accuracy
  const freqHz = (peakBin * fps) / fftSize;
  return Math.round(freqHz * 60); // Convert Hz to BPM
}

function findPeaks(signal: number[], minDistance: number): number[] {
  const peaks: number[] = [];
  for (let i = 2; i < signal.length - 2; i++) {
    if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] &&
        signal[i] > signal[i - 2] && signal[i] > signal[i + 2]) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistance) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

function analyzePPG(rawSignal: number[], fps: number): PulseResult {
  // Discard first 2 seconds (settling time)
  const settleFrames = Math.min(Math.round(fps * 2), Math.floor(rawSignal.length * 0.1));
  const signal = rawSignal.slice(settleFrames);

  // Detrend (remove slow drift)
  const detrended = detrendSignal(signal, fps);

  // Smooth aggressively to remove high-freq noise
  const smoothed = smoothSignal(detrended, Math.round(fps / 6));

  // Primary HR via FFT (robust)
  const hr = fftHeartRate(smoothed, fps);

  // Secondary: peak detection on smoothed signal for HRV
  const minDist = Math.round(fps * (60 / Math.max(hr * 1.3, 100))); // based on FFT HR
  const peaks = findPeaks(smoothed, minDist);
  const rrIntervals: number[] = [];
  for (let i = 1; i < peaks.length; i++) {
    const rrMs = ((peaks[i] - peaks[i - 1]) / fps) * 1000;
    const expectedRR = 60000 / hr;
    // Only keep RR intervals within 40% of expected (filter outliers)
    if (rrMs > expectedRR * 0.6 && rrMs < expectedRR * 1.4) {
      rrIntervals.push(rrMs);
    }
  }

  // HRV (RMSSD) - only from filtered RR intervals
  let sumSqDiff = 0;
  let validDiffs = 0;
  for (let i = 1; i < rrIntervals.length; i++) {
    const diff = rrIntervals[i] - rrIntervals[i - 1];
    sumSqDiff += diff * diff;
    validDiffs++;
  }
  const hrv = validDiffs > 0 ? Math.round(Math.sqrt(sumSqDiff / validDiffs)) : 0;

  // Rhythm regularity from filtered RR intervals
  const meanRR = rrIntervals.length > 0 ? rrIntervals.reduce((a, b) => a + b, 0) / rrIntervals.length : (60000 / hr);
  const stdRR = rrIntervals.length > 1 ? Math.sqrt(rrIntervals.reduce((s, rr) => s + (rr - meanRR) ** 2, 0) / rrIntervals.length) : 0;
  const irregularity = meanRR > 0 ? stdRR / meanRR : 0;

  let rhythm: "regular" | "irregular" | "highly_irregular" = "regular";
  if (irregularity > 0.20) rhythm = "highly_irregular";
  else if (irregularity > 0.12) rhythm = "irregular";

  // Downsample smoothed signal for display
  const displayLen = 200;
  const step = Math.max(1, Math.floor(smoothed.length / displayLen));
  const displaySignal: number[] = [];
  for (let i = 0; i < smoothed.length; i += step) displaySignal.push(smoothed[i]);

  return { hr, hrv, rrIntervals, irregularity, rhythm, signal: displaySignal };
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
        {/* Grid lines */}
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
  const [liveHR, setLiveHR] = useState<number | null>(null);
  const [ppgWaveform, setPpgWaveform] = useState<number[]>([]);

  // Sound state
  const [soundState, setSoundState] = useState<SoundState>("idle");
  const [soundResult, setSoundResult] = useState<SoundResult | null>(null);
  const [soundCountdown, setSoundCountdown] = useState(RECORD_DURATION);
  const [soundWaveform, setSoundWaveform] = useState<number[]>([]);
  // Live mic sensitivity (user-adjustable while placing the phone) + live beat feedback.
  // A blind fixed gain just amplifies the noise floor ("radio static"); letting the user
  // tune it against a live heart-band meter is how the working apps (Echoes) hit ~80%.
  const [micGain, setMicGain] = useState(12);
  const [beatLevel, setBeatLevel] = useState(0); // 0-1, live heart-band energy
  const [liveBeatBpm, setLiveBeatBpm] = useState<number | null>(null);
  const [beatPulse, setBeatPulse] = useState(0); // increments each detected beat (drives ♥ animation)

  // Refs
  const modelRef = useRef<tf.LayersModel | null>(null);
  const normRef = useRef<{ mean: number[]; std: number[] } | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ppgSignalRef = useRef<number[]>([]);
  const ppgFilteredRef = useRef<number[]>([]);
  const ppgTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const ppgAnimRef = useRef<number>(0);
  const ppgStreamRef = useRef<MediaStream | null>(null);
  const pcmBufferRef = useRef<Float32Array[]>([]);
  const scriptNodeRef = useRef<ScriptProcessorNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null); // live-adjustable capture gain
  const beatEnvRef = useRef<{ baseline: number; lastBeatT: number; beatTimes: number[] }>({ baseline: 0, lastBeatT: 0, beatTimes: [] });
  const soundTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundAnimRef = useRef<number>(0);
  const soundCtxRef = useRef<AudioContext | null>(null);
  const soundStreamRef = useRef<MediaStream | null>(null);
  const nativeSampleRateRef = useRef<number>(44100);
  const filteredAudioRef = useRef<Float32Array | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const playbackSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

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
    ppgSignalRef.current = [];
    ppgFilteredRef.current = [];
    setPpgWaveform([]);
    setLiveHR(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 30 } },
      });
      ppgStreamRef.current = stream;

      const track = stream.getVideoTracks()[0];
      try { await track.applyConstraints({ advanced: [{ torch: true } as MediaTrackConstraintSet] }); } catch {}

      setPulseState("measuring");
      await new Promise(r => setTimeout(r, 150));

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try { await videoRef.current.play(); } catch {}
      }

      setPulseCountdown(PPG_DURATION);

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d", { willReadFrequently: true })!;

      const captureFrame = () => {
        const video = videoRef.current;
        if (video && video.readyState >= 2) {
          canvas.width = video.videoWidth || 320;
          canvas.height = video.videoHeight || 240;
          ctx.drawImage(video, 0, 0);
          const cx = Math.floor(canvas.width / 2);
          const cy = Math.floor(canvas.height / 2);
          const size = Math.min(50, Math.floor(canvas.width / 4));
          const imageData = ctx.getImageData(Math.max(0, cx - size), Math.max(0, cy - size), size * 2, size * 2);
          const data = imageData.data;
          let rSum = 0, gSum = 0, count = 0;
          for (let i = 0; i < data.length; i += 4) { rSum += data[i]; gSum += data[i + 1]; count++; }
          const rAvg = rSum / count;
          const gAvg = gSum / count;

          setFingerDetected(rAvg > 30 || rAvg > gAvg);
          ppgSignalRef.current.push(rAvg);

          // Real-time bandpass for waveform display
          const sig = ppgSignalRef.current;
          if (sig.length > 10) {
            // Simple real-time high-pass: subtract moving average
            const winSize = Math.min(sig.length, Math.round(PPG_FPS * 1.5));
            const recent = sig.slice(-winSize);
            const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
            const filtered = rAvg - mean;
            ppgFilteredRef.current.push(filtered);
            setPpgWaveform([...ppgFilteredRef.current]);
          }

          // Live HR via FFT every 2 seconds (use last 10 seconds of data)
          if (sig.length > PPG_FPS * 5 && sig.length % (PPG_FPS * 2) < 2) {
            try {
              const window = sig.slice(-PPG_FPS * 10);
              const detrended = detrendSignal(window, PPG_FPS);
              const smoothed = smoothSignal(detrended, Math.round(PPG_FPS / 6));
              const liveRate = fftHeartRate(smoothed, PPG_FPS);
              if (liveRate >= 45 && liveRate <= 180) setLiveHR(liveRate);
            } catch {}
          }
        }
        ppgAnimRef.current = requestAnimationFrame(captureFrame);
      };
      captureFrame();

      let remaining = PPG_DURATION;
      ppgTimerRef.current = setInterval(() => {
        remaining--;
        setPulseCountdown(remaining);
        if (remaining <= 0) stopPPG();
      }, 1000);
    } catch {
      setError("Camera access denied. Please allow camera access and try again.");
    }
  }, []);

  const stopPPG = useCallback(() => {
    if (ppgTimerRef.current) clearInterval(ppgTimerRef.current);
    cancelAnimationFrame(ppgAnimRef.current);
    if (ppgStreamRef.current) ppgStreamRef.current.getTracks().forEach(t => t.stop());

    setPulseState("processing");
    const signal = ppgSignalRef.current;
    if (signal.length < PPG_FPS * 5) {
      setError("Not enough data. Keep finger on camera for at least 5 seconds.");
      setPulseState("idle");
      return;
    }
    try {
      const result = analyzePPG(signal, PPG_FPS);
      if (result.hr < 30 || result.hr > 220) {
        setError("Could not detect valid pulse. Try again with finger firmly over camera.");
        setPulseState("idle");
        return;
      }
      setPulseResult(result);
      setPulseState("done");
    } catch {
      setError("Analysis failed. Try again.");
      setPulseState("idle");
    }
  }, []);

  // ============ SOUND LOGIC ============
  const startSound = useCallback(async () => {
    setError(null);
    setSoundResult(null);
    pcmBufferRef.current = [];
    setSoundWaveform([]);

    try {
      // Disable ALL of the phone's built-in voice processing. Standard flags
      // (echoCancellation/noiseSuppression/autoGainControl) plus Chrome/Android's
      // non-standard "goog*" flags — critically googHighpassFilter, which Android
      // applies by default and which chops off the 20-200Hz band where heart sounds
      // actually live. Without turning it off, the heartbeat never reaches us on Android.
      const audioConstraints = {
        sampleRate: { ideal: 44100 },
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        googEchoCancellation: false,
        googAutoGainControl: false,
        googNoiseSuppression: false,
        googHighpassFilter: false,
        googTypingNoiseDetection: false,
        googAudioMirroring: false,
        voiceIsolation: false,
      } as MediaTrackConstraints;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

      // RAW PCM capture — no codec, no lossy compression
      // Heart sounds live at 20-150Hz; Opus codec destroys these frequencies.
      // We capture uncompressed samples and do ALL filtering offline in DSP.
      const audioCtx = new AudioContext();
      soundCtxRef.current = audioCtx;
      // Mobile browsers (iOS Safari especially) start the AudioContext SUSPENDED.
      // A suspended context never fires onaudioprocess, so the PCM buffer stays empty
      // and nothing gets recorded. Resume it inside this user-gesture-initiated call.
      if (audioCtx.state === "suspended") {
        try { await audioCtx.resume(); } catch {}
      }
      nativeSampleRateRef.current = audioCtx.sampleRate; // Store actual rate (48000 on most phones)
      const source = audioCtx.createMediaStreamSource(stream);

      // CAPTURE branch: raw, ungained PCM. We do NOT apply the user's gain here —
      // gaining before capture risks clipping the samples, and the offline pipeline
      // normalizes anyway. Raw capture preserves the cleanest possible signal.
      const bufferSize = 4096;
      const scriptNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      scriptNodeRef.current = scriptNode;
      scriptNode.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        pcmBufferRef.current.push(new Float32Array(inputData));
        e.outputBuffer.getChannelData(0).fill(0); // silence output, no speaker feedback
      };
      source.connect(scriptNode);
      // ScriptProcessorNode must reach destination to fire onaudioprocess (output silenced above)
      scriptNode.connect(audioCtx.destination);

      // MONITOR branch (drives the live meter + beat detector, NOT the capture):
      // source → live gain → 20-200Hz bandpass → analyser. The user drags the gain
      // slider until the heart-band meter shows a rhythmic pulse, which tells them the
      // phone is positioned right BEFORE the 20s recording is spent.
      const gainNode = audioCtx.createGain();
      gainNode.gain.value = micGain;
      gainNodeRef.current = gainNode;
      const bandpass = audioCtx.createBiquadFilter();
      bandpass.type = "bandpass";
      bandpass.frequency.value = 70;   // center of S1/S2 band
      bandpass.Q.value = 0.7;          // wide-ish to span ~20-200Hz
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(gainNode);
      gainNode.connect(bandpass);
      bandpass.connect(analyser);
      // Pull the monitor branch to destination through a muted gain so the graph runs
      // (an analyser not connected downstream may not be processed in some browsers).
      const muteGain = audioCtx.createGain();
      muteGain.gain.value = 0;
      analyser.connect(muteGain);
      muteGain.connect(audioCtx.destination);

      // reset live beat tracker
      beatEnvRef.current = { baseline: 0, lastBeatT: 0, beatTimes: [] };
      setBeatLevel(0);
      setLiveBeatBpm(null);

      // Live waveform + heart-band level meter + beat detection
      const waveData = new Float32Array(analyser.fftSize);
      const updateWaveform = () => {
        analyser.getFloatTimeDomainData(waveData);
        const step = Math.floor(waveData.length / 50);
        const points: number[] = [];
        let sumSq = 0;
        for (let i = 0; i < waveData.length; i++) sumSq += waveData[i] * waveData[i];
        for (let i = 0; i < waveData.length; i += step) points.push(waveData[i]);
        const rms = Math.sqrt(sumSq / waveData.length);

        // Meter: map rms (already gained) to 0-1 with a soft ceiling
        const level = Math.min(1, rms * 3);
        setBeatLevel(level);

        // Beat detection: adaptive-baseline threshold crossing with 300ms refractory
        const env = beatEnvRef.current;
        env.baseline = env.baseline * 0.95 + rms * 0.05; // slow-moving noise floor
        const t = audioCtx.currentTime;
        if (rms > env.baseline * 2.2 && rms > 0.02 && t - env.lastBeatT > 0.3) {
          env.lastBeatT = t;
          env.beatTimes.push(t);
          if (env.beatTimes.length > 8) env.beatTimes.shift();
          setBeatPulse(p => p + 1);
          if (env.beatTimes.length >= 4) {
            const intervals: number[] = [];
            for (let i = 1; i < env.beatTimes.length; i++) intervals.push(env.beatTimes[i] - env.beatTimes[i - 1]);
            intervals.sort((a, b) => a - b);
            const medInt = intervals[Math.floor(intervals.length / 2)];
            const bpm = medInt > 0 ? Math.round(60 / medInt) : 0;
            if (bpm >= 40 && bpm <= 200) setLiveBeatBpm(bpm);
          }
        }

        setSoundWaveform(prev => {
          const next = [...prev, ...points];
          return next.slice(-WAVEFORM_POINTS * 2);
        });
        soundAnimRef.current = requestAnimationFrame(updateWaveform);
      };
      updateWaveform();

      // Store stream ref for cleanup
      soundStreamRef.current = stream;

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
    }
  }, [micGain]);

  const stopSound = useCallback(async () => {
    if (soundTimerRef.current) clearInterval(soundTimerRef.current);
    cancelAnimationFrame(soundAnimRef.current);
    gainNodeRef.current = null;
    setBeatLevel(0);
    // Disconnect and clean up ScriptProcessorNode
    if (scriptNodeRef.current) {
      scriptNodeRef.current.disconnect();
      scriptNodeRef.current = null;
    }
    if (soundCtxRef.current) { soundCtxRef.current.close(); soundCtxRef.current = null; }
    // Stop mic stream
    if (soundStreamRef.current) {
      soundStreamRef.current.getTracks().forEach(t => t.stop());
      soundStreamRef.current = null;
    }
    // Process captured PCM
    await processSound();
  }, []);

  const playFilteredAudio = useCallback(() => {
    const audio = filteredAudioRef.current;
    if (!audio) return;

    // Stop if already playing
    if (isPlaying && playbackSourceRef.current) {
      playbackSourceRef.current.stop();
      playbackSourceRef.current = null;
      if (playbackCtxRef.current) { playbackCtxRef.current.close(); playbackCtxRef.current = null; }
      setIsPlaying(false);
      return;
    }

    // Play the filtered audio at a higher sample rate for audibility
    // Heart sounds at 2000Hz SR are very low frequency — upsample to 8000Hz to make them audible
    const playbackSR = 8000;
    const upRatio = playbackSR / SAMPLE_RATE;
    const upLen = Math.floor(audio.length * upRatio);

    const ctx = new AudioContext({ sampleRate: playbackSR });
    playbackCtxRef.current = ctx;
    const buffer = ctx.createBuffer(1, upLen, playbackSR);
    const channelData = buffer.getChannelData(0);

    // Upsample with linear interpolation
    for (let i = 0; i < upLen; i++) {
      const srcPos = i / upRatio;
      const idx = Math.floor(srcPos);
      const frac = srcPos - idx;
      const s0 = idx < audio.length ? audio[idx] : 0;
      const s1 = idx + 1 < audio.length ? audio[idx + 1] : s0;
      channelData[i] = s0 + frac * (s1 - s0);
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => {
      setIsPlaying(false);
      playbackSourceRef.current = null;
      ctx.close();
      playbackCtxRef.current = null;
    };
    source.start();
    playbackSourceRef.current = source;
    setIsPlaying(true);
  }, [isPlaying]);

  const processSound = useCallback(async () => {
    setSoundState("processing");
    try {
      // Concatenate raw PCM chunks into a single Float32Array
      const chunks = pcmBufferRef.current;
      if (chunks.length === 0) {
        setError("No audio captured.");
        setSoundState("idle");
        return;
      }
      const totalSamples = chunks.reduce((sum, c) => sum + c.length, 0);
      const rawData = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of chunks) {
        rawData.set(chunk, offset);
        offset += chunk.length;
      }
      pcmBufferRef.current = []; // free memory

      // Raw audio is at AudioContext sample rate (typically 44100 or 48000Hz)
      // Resample to SAMPLE_RATE (2000Hz) for DSP processing
      const nativeSR = nativeSampleRateRef.current;
      const ratio = nativeSR / SAMPLE_RATE;
      const totalResampled = Math.floor(rawData.length / ratio);
      const resampled = new Float32Array(totalResampled);
      for (let i = 0; i < totalResampled; i++) {
        const srcIdx = Math.floor(i * ratio);
        resampled[i] = srcIdx < rawData.length ? rawData[srcIdx] : 0;
      }

      // Normalize resampled audio to [-1, 1] — raw PCM with 10x gain may clip
      let peakVal = 0;
      for (let i = 0; i < resampled.length; i++) {
        const v = Math.abs(resampled[i]);
        if (v > peakVal) peakVal = v;
      }
      if (peakVal > 1.0) {
        const scale = 0.95 / peakVal;
        for (let i = 0; i < resampled.length; i++) resampled[i] *= scale;
      }

      // Quality assessment on the full recording
      const qualityReport = assessSignalQuality(resampled, SAMPLE_RATE);

      // If quality is too low, skip ML analysis entirely — results would be meaningless
      if (!qualityReport.pass) {
        // Still run noise reduction for playback
        const { cleaned: cleanedAudio } = cleanHeartSound(resampled, SAMPLE_RATE);
        const playbackAudio = new Float32Array(cleanedAudio.length);
        let maxA = 0;
        for (let i = 0; i < cleanedAudio.length; i++) { const v = Math.abs(cleanedAudio[i]); if (v > maxA) maxA = v; }
        const bf = maxA > 0 ? 0.8 / maxA : 1;
        for (let i = 0; i < cleanedAudio.length; i++) playbackAudio[i] = cleanedAudio[i] * bf;
        filteredAudioRef.current = playbackAudio;

        setSoundResult({
          label: "low_quality",
          confidence: 0,
          normal: 0,
          abnormal: 0,
          segmentsAnalyzed: 0,
          bestSegmentIdx: 0,
          heartSoundFindings: null,
          qualityReport,
        });
        setSoundState("done");
        return;
      }

      // Score each 3-second window by "quality" (moderate energy, low variance = clean heartbeat)
      const windowSize = SAMPLE_RATE * SEGMENT_DURATION;
      const hopSize = Math.floor(windowSize / 2);
      interface WindowScore { start: number; score: number; rms: number }
      const windowScores: WindowScore[] = [];

      for (let start = 0; start + windowSize <= resampled.length; start += hopSize) {
        const segment = resampled.slice(start, start + windowSize);
        let rms = 0;
        for (let i = 0; i < segment.length; i++) rms += segment[i] * segment[i];
        rms = Math.sqrt(rms / segment.length);
        if (rms < 0.0005) continue; // skip near-silence

        // Calculate energy variance (lower = more consistent = cleaner heartbeat)
        const chunkSize = Math.floor(segment.length / 10);
        const chunkEnergies: number[] = [];
        for (let c = 0; c < 10; c++) {
          let e = 0;
          for (let i = c * chunkSize; i < (c + 1) * chunkSize; i++) e += segment[i] * segment[i];
          chunkEnergies.push(e / chunkSize);
        }
        const meanEnergy = chunkEnergies.reduce((a, b) => a + b, 0) / chunkEnergies.length;
        const variance = chunkEnergies.reduce((s, e) => s + (e - meanEnergy) ** 2, 0) / chunkEnergies.length;
        const cv = Math.sqrt(variance) / (meanEnergy + 1e-10); // coefficient of variation

        // Score: prefer moderate RMS and low CV (consistent energy = clean heartbeat pattern)
        const rmsScore = rms > 0.001 && rms < 0.5 ? 1 : 0.3;
        const cvScore = Math.max(0, 1 - cv * 2); // lower CV = higher score
        windowScores.push({ start, score: rmsScore * cvScore, rms });
      }

      if (windowScores.length === 0) {
        setError("No usable audio detected.");
        setSoundState("idle");
        return;
      }

      // Sort by quality score, take top 5 best segments
      windowScores.sort((a, b) => b.score - a.score);
      const bestWindows = windowScores.slice(0, Math.min(5, windowScores.length));
      const norm = normRef.current!;
      const segmentResults: { normal: number; abnormal: number; score: number }[] = [];

      for (const win of bestWindows) {
        const segment = resampled.slice(win.start, win.start + windowSize);
        const mfccFrames = extractMfcc(segment);
        const paddedFrames: Float32Array[] = [];
        for (let i = 0; i < MAX_FRAMES; i++) paddedFrames.push(i < mfccFrames.length ? mfccFrames[i] : new Float32Array(N_MFCC));

        const inputData = new Float32Array(MAX_FRAMES * N_MFCC);
        for (let i = 0; i < MAX_FRAMES; i++) for (let j = 0; j < N_MFCC; j++) inputData[i * N_MFCC + j] = (paddedFrames[i][j] - norm.mean[j]) / (norm.std[j] + 1e-8);

        const inputTensor = tf.tensor3d(inputData, [1, MAX_FRAMES, N_MFCC]);
        const prediction = modelRef.current!.predict(inputTensor) as tf.Tensor;
        const probs = await prediction.data();
        inputTensor.dispose();
        prediction.dispose();
        const pNormal = isNaN(probs[0]) ? 0.5 : probs[0];
        const pAbnormal = isNaN(probs[1]) ? 0.5 : probs[1];
        segmentResults.push({ normal: pNormal, abnormal: pAbnormal, score: win.score });
      }

      // Weighted average by quality score
      let totalWeight = 0;
      let wNormal = 0, wAbnormal = 0;
      for (const r of segmentResults) {
        wNormal += r.normal * r.score;
        wAbnormal += r.abnormal * r.score;
        totalWeight += r.score;
      }
      const avgNormal = totalWeight > 0 ? wNormal / totalWeight : 0.5;
      const avgAbnormal = totalWeight > 0 ? wAbnormal / totalWeight : 0.5;

      // DSP-based heart sound analysis on the best quality segment
      const bestWin = bestWindows[0];
      // Analyze a longer segment for better S1/S2 detection (use up to 10 seconds)
      const longWindowSize = Math.min(SAMPLE_RATE * 10, resampled.length);
      const longSegment = resampled.slice(bestWin.start, Math.min(bestWin.start + longWindowSize, resampled.length));
      const heartSoundFindings = analyzeHeartSounds(longSegment, SAMPLE_RATE);

      // If DSP detects serious findings, adjust the label
      let finalLabel = avgAbnormal > avgNormal ? "abnormal" : "normal";
      if (heartSoundFindings.s3Suspected || heartSoundFindings.s4Suspected ||
          heartSoundFindings.systolicMurmur || heartSoundFindings.diastolicMurmur) {
        finalLabel = "abnormal";
      }

      // Store CLEANED audio for playback — run full noise reduction pipeline
      const { cleaned: cleanedAudio, snrImprovement: snrGain } = cleanHeartSound(resampled, SAMPLE_RATE);
      const playbackAudio = new Float32Array(cleanedAudio.length);
      let maxAmp = 0;
      for (let i = 0; i < cleanedAudio.length; i++) {
        const v = Math.abs(cleanedAudio[i]);
        if (v > maxAmp) maxAmp = v;
      }
      const boostFactor = maxAmp > 0 ? 0.8 / maxAmp : 1; // normalize to 80% max
      for (let i = 0; i < cleanedAudio.length; i++) {
        playbackAudio[i] = cleanedAudio[i] * boostFactor;
      }
      filteredAudioRef.current = playbackAudio;

      // If noise reduction couldn't extract any signal (negative SNR gain)
      // AND DSP didn't detect S1/S2, the recording has no heart sounds — override ML
      if (snrGain < 0.5 && !heartSoundFindings.s1s2Detected) {
        finalLabel = "low_quality";
        qualityReport.pass = false;
        if (!qualityReport.issues.includes("No heart sounds detected in signal")) {
          qualityReport.issues.push("No heart sounds detected in signal");
          qualityReport.suggestions.push("Position mic directly on bare chest, left side between ribs");
        }
      }

      setSoundResult({
        label: finalLabel,
        confidence: finalLabel === "low_quality" ? 0 : Math.max(avgNormal, avgAbnormal),
        normal: finalLabel === "low_quality" ? 0 : avgNormal,
        abnormal: finalLabel === "low_quality" ? 0 : avgAbnormal,
        segmentsAnalyzed: segmentResults.length,
        bestSegmentIdx: 0,
        heartSoundFindings: finalLabel === "low_quality" ? null : heartSoundFindings,
        qualityReport,
      });
      setSoundState("done");
    } catch (e) {
      console.error(e);
      setError("Audio processing failed.");
      setSoundState("idle");
    }
  }, []);

  // ============ RENDER ============
  const tabClass = (t: Tab) => `flex-1 py-2 text-sm font-medium rounded-lg transition-colors ${tab === t ? "bg-slate-700 text-white" : "text-slate-400 hover:text-slate-300"}`;

  return (
    <div className="flex flex-col items-center min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-white p-4">
      <div className="mt-4 mb-3 text-center">
        <h1 className="text-2xl font-bold tracking-tight">CardioListen</h1>
        <p className="text-slate-400 text-xs mt-0.5">AI Cardiac Health Screen</p>
      </div>

      <div className="w-full max-w-sm flex gap-1 bg-slate-800/80 p-1 rounded-xl mb-3">
        <button className={tabClass("pulse")} onClick={() => setTab("pulse")}>Pulse</button>
        <button className={tabClass("sound")} onClick={() => setTab("sound")}>Heart Sound</button>
        <button className={tabClass("report")} onClick={() => setTab("report")}>Report</button>
      </div>

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
                  <p className="text-slate-500 text-xs mt-1">Cover rear camera with fingertip, flash will turn on</p>
                </div>
                <button onClick={startPPG} className="w-full py-3 bg-rose-500 hover:bg-rose-600 rounded-xl font-semibold transition-colors">
                  Start Pulse Reading
                </button>
              </div>
            )}

            {pulseState === "measuring" && (
              <div className="flex flex-col gap-3">
                {/* Top bar: camera preview + countdown + HR */}
                <div className="flex items-center gap-3">
                  <div className="relative flex-shrink-0">
                    <video ref={videoRef} className="w-16 h-16 rounded-full object-cover border-2 border-rose-500" muted playsInline autoPlay />
                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                  <div className="flex-1">
                    <div className={`text-xs font-medium ${fingerDetected ? "text-green-400" : "text-yellow-400 animate-pulse"}`}>
                      {fingerDetected ? "Finger detected" : "Place finger on camera"}
                    </div>
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

                {/* Live ECG-style waveform */}
                <LiveWaveform data={ppgWaveform} color="#f43f5e" height={120} />

                {/* Progress bar */}
                <div className="w-full bg-slate-700 rounded-full h-1">
                  <div className="bg-rose-500 h-1 rounded-full transition-all" style={{ width: `${((PPG_DURATION - pulseCountdown) / PPG_DURATION) * 100}%` }} />
                </div>

                <button onClick={stopPPG} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Stop & Analyze
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
                {/* Final waveform */}
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
                  <p>HR {pulseResult.hr >= 60 && pulseResult.hr <= 100 ? "is within" : pulseResult.hr < 60 ? "is below" : "is above"} normal resting range (60-100)</p>
                  <p>HRV {pulseResult.hrv > 50 ? "suggests good" : pulseResult.hrv > 20 ? "is moderate" : "is low"} autonomic health</p>
                  {pulseResult.rhythm !== "regular" && <p className="text-yellow-400">Irregular rhythm detected</p>}
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
                <div className="text-center space-y-2">
                  <p className="text-slate-300 text-sm font-medium">Heart Sound Analysis</p>
                  <div className="bg-slate-700/30 rounded-lg p-3 text-left space-y-1.5">
                    <p className="text-slate-400 text-xs font-medium">Recording tips:</p>
                    <p className="text-slate-500 text-xs">1. Remove phone case</p>
                    <p className="text-slate-500 text-xs">2. Find your phone&apos;s mic (bottom edge, near charging port)</p>
                    <p className="text-slate-500 text-xs">3. Press mic firmly on bare skin, left of breastbone, between ribs</p>
                    <p className="text-slate-500 text-xs">4. Breathe out, then hold your breath while it records</p>
                    <p className="text-slate-500 text-xs">5. Quiet room &mdash; turn off fans/AC</p>
                    <p className="text-slate-500 text-xs">6. Drag the sensitivity slider until the beat meter pulses</p>
                  </div>
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
                    <p className="text-slate-500 text-xs">AI noise reduction active</p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-lg font-bold">{soundCountdown}</span>
                  </div>
                </div>

                {/* Live heart sound waveform */}
                <LiveWaveform data={soundWaveform} color="#3b82f6" height={140} />

                {/* Live heart-band meter + beat feedback */}
                <div className="bg-slate-800/60 rounded-xl p-3 flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-slate-400">Heart-band signal (20–200Hz)</span>
                    {liveBeatBpm ? (
                      <span
                        key={beatPulse}
                        className="text-xs font-semibold text-rose-400 flex items-center gap-1 animate-pulse"
                      >
                        ♥ ~{liveBeatBpm} bpm
                      </span>
                    ) : (
                      <span className="text-xs text-slate-500">move phone / raise gain…</span>
                    )}
                  </div>
                  <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                    <div
                      className={`h-3 rounded-full transition-[width] duration-75 ${beatLevel > 0.5 ? "bg-rose-500" : beatLevel > 0.2 ? "bg-amber-400" : "bg-blue-500"}`}
                      style={{ width: `${Math.round(beatLevel * 100)}%` }}
                    />
                  </div>
                  <p className="text-[11px] leading-tight text-slate-500">
                    {liveBeatBpm
                      ? "Good — a rhythmic pulse is showing. Hold the phone dead still here."
                      : "Watch this bar pulse in rhythm with your heart. If it's flat or just jittery noise, reposition the phone and adjust gain below."}
                  </p>
                </div>

                {/* Live mic sensitivity (gain) slider */}
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Mic sensitivity</span>
                    <span className="tabular-nums text-slate-300">{micGain}×</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={30}
                    step={1}
                    value={micGain}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setMicGain(v);
                      if (gainNodeRef.current) gainNodeRef.current.gain.value = v;
                    }}
                    className="w-full accent-rose-500"
                  />
                  <p className="text-[11px] text-slate-500">Turn up until the bar reacts to your heartbeat; turn down if it&apos;s maxed out / just noise.</p>
                </div>

                <div className="w-full bg-slate-700 rounded-full h-1">
                  <div className="bg-blue-500 h-1 rounded-full transition-all" style={{ width: `${((RECORD_DURATION - soundCountdown) / RECORD_DURATION) * 100}%` }} />
                </div>

                <button onClick={() => stopSound()} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Stop & Analyze
                </button>
              </div>
            )}

            {soundState === "processing" && (
              <div className="flex flex-col items-center gap-4 py-8">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-blue-400 text-sm">Cleaning signal + AI analysis...</p>
                <p className="text-slate-500 text-xs">Noise reduction → heart sound extraction → classification</p>
              </div>
            )}

            {soundState === "done" && soundResult && (
              <div className="flex flex-col items-center gap-3">
                {/* LOW QUALITY — show retry guidance instead of unreliable results */}
                {soundResult.label === "low_quality" ? (
                  <>
                    <div className="w-14 h-14 rounded-full flex items-center justify-center bg-slate-500/10 border-2 border-slate-500">
                      <svg className="w-7 h-7 text-slate-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" /></svg>
                    </div>
                    <div className="text-center">
                      <h2 className="text-xl font-bold text-slate-400">Recording Quality Too Low</h2>
                      <p className="text-slate-500 text-xs mt-1">Could not reliably analyze — please try again</p>
                      {soundResult.qualityReport && <p className="text-red-400 text-xs mt-1">Quality Score: {soundResult.qualityReport.score}%</p>}
                    </div>
                    {soundResult.qualityReport && (
                      <div className="w-full space-y-1.5">
                        <div className="text-xs text-slate-400 font-medium">What to fix:</div>
                        {soundResult.qualityReport.issues.map((issue, i) => (
                          <div key={`i${i}`} className="text-xs text-red-300 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{issue}</div>
                        ))}
                        {soundResult.qualityReport.suggestions.map((tip, i) => (
                          <div key={`s${i}`} className="text-xs text-slate-400 bg-slate-700/30 rounded-lg px-3 py-2">{tip}</div>
                        ))}
                      </div>
                    )}
                    <p className="text-slate-500 text-xs text-center">Tap "Play Heart Sound" to hear what the mic captured</p>
                  </>
                ) : (
                  <>
                <div className={`w-14 h-14 rounded-full flex items-center justify-center ${soundResult.label === "normal" ? "bg-green-500/10 border-2 border-green-500" : "bg-yellow-500/10 border-2 border-yellow-500"}`}>
                  {soundResult.label === "normal" ? (
                    <svg className="w-7 h-7 text-green-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                  ) : (
                    <svg className="w-7 h-7 text-yellow-400" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" /></svg>
                  )}
                </div>
                <div className="text-center">
                  <h2 className={`text-xl font-bold ${soundResult.label === "normal" ? "text-green-400" : "text-yellow-400"}`}>
                    {soundResult.label === "normal" ? "Normal Heart Sound" : "Potential Abnormality"}
                  </h2>
                  <p className="text-slate-400 text-xs mt-1">Confidence: {(soundResult.confidence * 100).toFixed(1)}%</p>
                  <p className="text-slate-500 text-xs">Best {soundResult.segmentsAnalyzed} of {Math.floor(RECORD_DURATION / SEGMENT_DURATION)} segments used</p>
                </div>
                <div className="w-full space-y-2">
                  <div>
                    <div className="flex justify-between text-xs mb-1"><span className="text-green-400">Normal</span><span className="text-slate-400">{(soundResult.normal * 100).toFixed(1)}%</span></div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden"><div className="h-full bg-green-500 rounded-full" style={{ width: `${soundResult.normal * 100}%` }} /></div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs mb-1"><span className="text-yellow-400">Abnormal</span><span className="text-slate-400">{(soundResult.abnormal * 100).toFixed(1)}%</span></div>
                    <div className="h-2 bg-slate-700 rounded-full overflow-hidden"><div className="h-full bg-yellow-500 rounded-full" style={{ width: `${soundResult.abnormal * 100}%` }} /></div>
                  </div>
                </div>
                {soundResult.heartSoundFindings && soundResult.heartSoundFindings.findings.length > 0 && (
                  <div className="w-full mt-2">
                    <div className="text-xs text-slate-400 font-medium mb-2">Detailed Findings</div>
                    <div className="space-y-2">
                      {soundResult.heartSoundFindings.s3Suspected && (
                        <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-2">
                          <span className="text-red-400 text-sm mt-0.5">!</span>
                          <span className="text-red-300 text-xs">S3 Gallop &mdash; suggests heart failure / volume overload</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings.s4Suspected && (
                        <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/30 rounded-lg p-2">
                          <span className="text-red-400 text-sm mt-0.5">!</span>
                          <span className="text-red-300 text-xs">S4 Gallop &mdash; suggests stiff ventricle / acute MI</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings.systolicMurmur && (
                        <div className="flex items-start gap-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-2">
                          <span className="text-yellow-400 text-sm mt-0.5">!</span>
                          <span className="text-yellow-300 text-xs">Systolic murmur &mdash; may indicate valve disease or post-MI complication</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings.diastolicMurmur && (
                        <div className="flex items-start gap-2 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-2">
                          <span className="text-yellow-400 text-sm mt-0.5">!</span>
                          <span className="text-yellow-300 text-xs">Diastolic murmur &mdash; may indicate valve insufficiency</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings.heartRate > 0 && (
                        <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                          <span className="text-xs text-slate-400">Auscultation HR</span>
                          <span className="text-sm font-bold text-slate-300">{soundResult.heartSoundFindings.heartRate} BPM</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings.findings.filter(f => !f.startsWith("Heart rate from")).map((finding, i) => (
                        <div key={i} className="text-xs text-slate-400 bg-slate-700/20 rounded-lg px-3 py-2">{finding}</div>
                      ))}
                    </div>
                  </div>
                )}
                {soundResult.qualityReport && soundResult.qualityReport.score < 80 && (
                  <div className="w-full mt-2">
                    <div className="text-xs text-slate-400 font-medium mb-1">
                      Recording Quality: <span className={soundResult.qualityReport.score >= 60 ? "text-yellow-400" : "text-red-400"}>{soundResult.qualityReport.score}%</span>
                    </div>
                    {soundResult.qualityReport.suggestions.map((s, i) => (
                      <div key={i} className="text-xs text-slate-500 bg-slate-700/20 rounded px-2 py-1 mt-1">Tip: {s}</div>
                    ))}
                  </div>
                )}
                  </>
                )}
                <div className="flex gap-2 w-full">
                  <button onClick={playFilteredAudio} className={`flex-1 py-2 rounded-xl text-sm font-medium transition-colors ${isPlaying ? "bg-green-600 hover:bg-green-700 text-white" : "bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 border border-emerald-500/30"}`}>
                    {isPlaying ? "Stop Playback" : "Play Heart Sound"}
                  </button>
                  <button onClick={() => { setSoundState("idle"); setSoundResult(null); setSoundWaveform([]); filteredAudioRef.current = null; setIsPlaying(false); }} className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                    Record Again
                  </button>
                </div>
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
                  <div className="text-xs text-slate-500 mb-1">Overall Assessment</div>
                  {(() => {
                    let flags = 0, total = 0;
                    if (pulseResult) {
                      total += 3;
                      if (pulseResult.hr < 50 || pulseResult.hr > 110) flags++;
                      if (pulseResult.hrv < 20) flags++;
                      if (pulseResult.rhythm !== "regular") flags++;
                    }
                    if (soundResult) {
                      total += 4; // ML label + S3 + S4 + murmurs
                      if (soundResult.label === "abnormal") flags++;
                      if (soundResult.heartSoundFindings?.s3Suspected) flags++;
                      if (soundResult.heartSoundFindings?.s4Suspected) flags++;
                      if (soundResult.heartSoundFindings?.systolicMurmur || soundResult.heartSoundFindings?.diastolicMurmur) flags++;
                    }
                    const score = total > 0 ? Math.round(((total - flags) / total) * 100) : 0;
                    const color = score >= 80 ? "text-green-400" : score >= 50 ? "text-yellow-400" : "text-red-400";
                    const label = score >= 80 ? "Looking Good" : score >= 50 ? "Some Concerns" : "Needs Attention";
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
                    <>
                      <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                        <span className="text-xs text-slate-400">Heart Sound (AI)</span>
                        <span className={`text-sm font-bold ${soundResult.label === "normal" ? "text-green-400" : "text-yellow-400"}`}>{soundResult.label === "normal" ? "Normal" : "Atypical"} ({(soundResult.confidence * 100).toFixed(0)}%)</span>
                      </div>
                      {soundResult.heartSoundFindings?.s3Suspected && (
                        <div className="flex justify-between items-center bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                          <span className="text-xs text-red-300">S3 Gallop</span>
                          <span className="text-sm font-bold text-red-400">Suspected</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings?.s4Suspected && (
                        <div className="flex justify-between items-center bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                          <span className="text-xs text-red-300">S4 Gallop</span>
                          <span className="text-sm font-bold text-red-400">Suspected</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings?.systolicMurmur && (
                        <div className="flex justify-between items-center bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-2">
                          <span className="text-xs text-yellow-300">Systolic Murmur</span>
                          <span className="text-sm font-bold text-yellow-400">Detected</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings?.diastolicMurmur && (
                        <div className="flex justify-between items-center bg-yellow-500/10 border border-yellow-500/20 rounded-lg px-3 py-2">
                          <span className="text-xs text-yellow-300">Diastolic Murmur</span>
                          <span className="text-sm font-bold text-yellow-400">Detected</span>
                        </div>
                      )}
                      {soundResult.heartSoundFindings && soundResult.heartSoundFindings.heartRate > 0 && (
                        <div className="flex justify-between items-center bg-slate-700/30 rounded-lg px-3 py-2">
                          <span className="text-xs text-slate-400">Sound-based HR</span>
                          <span className="text-sm font-bold text-slate-300">{soundResult.heartSoundFindings.heartRate} BPM</span>
                        </div>
                      )}
                    </>
                  )}
                </div>
                {(!pulseResult || !soundResult) && (
                  <div className="text-center">
                    <p className="text-slate-500 text-xs mb-1">Complete all tests for full report</p>
                    {!pulseResult && <button onClick={() => setTab("pulse")} className="py-1 px-3 bg-rose-500/20 text-rose-400 rounded-lg text-xs mr-2">+ Pulse</button>}
                    {!soundResult && <button onClick={() => setTab("sound")} className="py-1 px-3 bg-blue-500/20 text-blue-400 rounded-lg text-xs">+ Sound</button>}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm max-w-sm text-center">
          {error}
          <button onClick={() => setError(null)} className="ml-2 underline text-xs">dismiss</button>
        </div>
      )}

      <div className="mt-4 max-w-sm text-center pb-6">
        <p className="text-slate-600 text-xs">For educational and wellness screening only. Not a medical device or diagnosis. A phone mic is not a stethoscope &mdash; treat results as a rough screen. If experiencing chest pain, fainting, or severe breathlessness, seek immediate medical care.</p>
      </div>
    </div>
  );
}
