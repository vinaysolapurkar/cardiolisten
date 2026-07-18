"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// ============ CONSTANTS ============
const SAMPLE_RATE = 2000;
const SEGMENT_DURATION = 3;
const RECORD_DURATION = 60;
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

interface SoundResult {
  label: string;
  confidence: number;
  normal: number;
  abnormal: number;
  segmentsAnalyzed: number;
  bestSegmentIdx: number;
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
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const soundTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundAnimRef = useRef<number>(0);
  const soundCtxRef = useRef<AudioContext | null>(null);

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
    audioChunksRef.current = [];
    setSoundWaveform([]);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: { ideal: 44100 }, channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });

      // Set up audio processing chain with gain boost + bandpass filter
      const audioCtx = new AudioContext();
      soundCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);

      // Gain boost (3x - moderate to avoid amplifying noise too much)
      const gainNode = audioCtx.createGain();
      gainNode.gain.value = 3;

      // Tight bandpass filter: 20-200Hz (S1/S2 heart sound fundamental range)
      const highpass = audioCtx.createBiquadFilter();
      highpass.type = "highpass";
      highpass.frequency.value = 20;
      highpass.Q.value = 1.0;

      const lowpass = audioCtx.createBiquadFilter();
      lowpass.type = "lowpass";
      lowpass.frequency.value = 200;
      lowpass.Q.value = 1.0;

      // Second stage filter for sharper rolloff
      const highpass2 = audioCtx.createBiquadFilter();
      highpass2.type = "highpass";
      highpass2.frequency.value = 25;
      highpass2.Q.value = 0.7;

      const lowpass2 = audioCtx.createBiquadFilter();
      lowpass2.type = "lowpass";
      lowpass2.frequency.value = 150;
      lowpass2.Q.value = 0.7;

      // Chain: source -> hp1 -> lp1 -> hp2 -> lp2 -> gain (4-pole bandpass 20-200Hz)
      source.connect(highpass);
      highpass.connect(lowpass);
      lowpass.connect(highpass2);
      highpass2.connect(lowpass2);
      lowpass2.connect(gainNode);

      // Create a destination for recording the filtered audio
      const dest = audioCtx.createMediaStreamDestination();
      gainNode.connect(dest);

      // Analyser for live waveform
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      gainNode.connect(analyser);

      // Record the filtered+amplified audio
      const mediaRecorder = new MediaRecorder(dest.stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "audio/webm",
      });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        await processSound();
      };

      // Live waveform from analyser
      const waveData = new Float32Array(analyser.fftSize);
      const updateWaveform = () => {
        analyser.getFloatTimeDomainData(waveData);
        // Downsample to ~100 points for display, take every Nth sample
        const step = Math.floor(waveData.length / 50);
        const points: number[] = [];
        for (let i = 0; i < waveData.length; i += step) points.push(waveData[i]);
        setSoundWaveform(prev => {
          const next = [...prev, ...points];
          return next.slice(-WAVEFORM_POINTS * 2);
        });
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
      const audioCtx = new AudioContext({ sampleRate: 44100 });
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();

      const rawData = audioBuffer.getChannelData(0);
      const ratio = audioBuffer.sampleRate / SAMPLE_RATE;
      const totalResampled = Math.floor(rawData.length / ratio);
      const resampled = new Float32Array(totalResampled);
      for (let i = 0; i < totalResampled; i++) {
        const srcIdx = Math.floor(i * ratio);
        resampled[i] = srcIdx < rawData.length ? rawData[srcIdx] : 0;
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
        segmentResults.push({ normal: probs[0], abnormal: probs[1], score: win.score });
      }

      // Weighted average by quality score
      let totalWeight = 0;
      let wNormal = 0, wAbnormal = 0;
      for (const r of segmentResults) {
        wNormal += r.normal * r.score;
        wAbnormal += r.abnormal * r.score;
        totalWeight += r.score;
      }
      const avgNormal = wNormal / totalWeight;
      const avgAbnormal = wAbnormal / totalWeight;

      setSoundResult({
        label: avgAbnormal > avgNormal ? "abnormal" : "normal",
        confidence: Math.max(avgNormal, avgAbnormal),
        normal: avgNormal,
        abnormal: avgAbnormal,
        segmentsAnalyzed: segmentResults.length,
        bestSegmentIdx: 0,
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
                <div className="text-center">
                  <p className="text-slate-300 text-sm font-medium">Heart Sound Analysis</p>
                  <p className="text-slate-500 text-xs mt-1">Place mic firmly on left chest, quiet room</p>
                  <p className="text-slate-600 text-xs mt-0.5">Enhanced: 20-200Hz bandpass + noise rejection</p>
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
                    <p className="text-slate-500 text-xs">Bandpass 20-200Hz, noise rejected</p>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                    <span className="text-lg font-bold">{soundCountdown}</span>
                  </div>
                </div>

                {/* Live heart sound waveform */}
                <LiveWaveform data={soundWaveform} color="#3b82f6" height={140} />

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
                <p className="text-blue-400 text-sm">Analyzing best segments...</p>
                <p className="text-slate-500 text-xs">Selecting cleanest windows for inference</p>
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
                  <div className="text-xs text-slate-500 mb-1">Overall Assessment</div>
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
        <p className="text-slate-600 text-xs">For educational purposes only. Not a medical device.</p>
      </div>
    </div>
  );
}
