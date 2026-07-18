# Guided Chest Placement + Live ECG-Style Waveform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Heart Sound tab's blind 30s recording with a guided search across 3 chest zones (10s quality check per zone, auto-advance on failure, extended capture on success), plus a real-time ECG-style beat-synced waveform instead of the raw jittery mic trace.

**Architecture:** All changes live in `pwa/app/page.tsx` (the existing single-file convention for this app — DSP helpers, state, and render are already co-located there; see `docs/superpowers/specs/2026-07-18-guided-auscultation-design.md` for why this plan doesn't introduce a new file-splitting convention). Two new pure, unit-tested functions extracted from/added alongside the existing DSP helpers (`scoreAudioWindow`/`scoreAudioWindows`/`bestWindowScore` for the pass/fail gate, `processBeatEnvelopeSample` for the live spike detector). The existing `startSound`/`stopSound`/`processSound` callbacks are replaced by a guided-flow state machine that keeps one microphone stream open across all zone attempts.

**Tech Stack:** Next.js 16 / React 19 / TypeScript, `@tensorflow/tfjs`, Web Audio API (`AudioContext`, `MediaRecorder`, `AnalyserNode`), Node `.mjs` scripts under `verify/` for regression tests (existing pattern — no test framework, plain assertions + `process.exit`).

## Global Constraints

- Heart Sound tab only. Do not modify the Pulse (camera) tab's state, callbacks, or render blocks.
- Do not modify `extractMfcc`, `resampleAudio`, or the TF.js model-call sequence — these are already verified against librosa (0.0000 max error, see `verify/compare.mjs`) and against the real model (see `verify/model_e2e.mjs`). Re-run both after this feature's changes to confirm no regression.
- Zone labels shown to the user must be plain language, no clinical terms ("Aortic", "Mitral", etc. are banned in UI copy) — per explicit user feedback ("they are not doctors").
- Pass/fail gate threshold is `score > 0.5` (exact, from the approved spec) — do not tune this without updating the spec.
- Keep the existing diagnostics logging pattern (`pushDiagLog`, the "Copy diagnostics" button) working for the new flow — every zone attempt (pass or fail) must produce one log line.
- The mic stream (`getUserMedia`) must be requested only once per guided-flow session (not once per zone) — re-prompting for permission per zone is explicitly out of scope per the approved spec.

---

### Task 1: Extract reusable window-quality scorer, regression test

**Files:**
- Modify: `pwa/app/page.tsx` (insert new functions after line 280, the end of `resampleAudio`; modify `processSound`'s inline scoring loop, current lines 894–922 and the `windowScores`/`totalWindows` references at lines 924–934 and 965/972/975)
- Create: `verify/window_score.mjs`

**Interfaces:**
- Produces: `interface WindowScore { score: number; rms: number }`, `interface ScoredWindow extends WindowScore { start: number }`, `function scoreAudioWindow(segment: Float32Array | Float64Array): WindowScore`, `function scoreAudioWindows(clip: Float32Array, sampleRate: number, windowSeconds: number, hopSeconds: number): { windows: ScoredWindow[]; totalWindows: number }`, `function bestWindowScore(clip: Float32Array, sampleRate: number, windowSeconds: number, hopSeconds: number): WindowScore`. Task 3 consumes `scoreAudioWindows` and `bestWindowScore`.

- [ ] **Step 1: Add the pure scoring functions to `pwa/app/page.tsx`**

Using Edit, find this exact block (end of `resampleAudio`, currently lines 274–282):

```tsx
  // remove DC offset
  let mean = 0;
  for (let i = 0; i < out.length; i++) mean += out[i];
  mean /= out.length || 1;
  for (let i = 0; i < out.length; i++) out[i] -= mean;
  return out;
}

// ============ PPG Signal Processing ============
```

Replace it with:

```tsx
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

// ============ PPG Signal Processing ============
```

- [ ] **Step 2: Refactor `processSound`'s inline scoring loop to use the new functions**

Using Edit, find this exact block (currently lines 894–934):

```tsx
      const windowSize = SAMPLE_RATE * SEGMENT_DURATION;
      const hopSize = Math.floor(windowSize / 2);
      interface WindowScore { start: number; score: number; rms: number }
      const windowScores: WindowScore[] = [];
      let totalWindows = 0;

      for (let start = 0; start + windowSize <= resampled.length; start += hopSize) {
        totalWindows++;
        const segment = resampled.subarray(start, start + windowSize);
        let rms = 0;
        for (let i = 0; i < segment.length; i++) rms += segment[i] * segment[i];
        rms = Math.sqrt(rms / segment.length);
        if (rms < 0.0005) continue;

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
        windowScores.push({ start, score: rmsScore * cvScore, rms });
      }

      if (windowScores.length === 0) {
        const msg = `${debugBase}, ${totalWindows} windows all below silence threshold (rms<0.0005) - mic likely picked up almost nothing`;
        setSoundDebug(msg);
        pushDiagLog(`SOUND FAIL: ${msg}`);
        setError("No usable audio detected. Record again in a quiet room with the mic pressed to your chest.");
        setSoundState("idle");
        return;
      }

      windowScores.sort((a, b) => b.score - a.score);
      const bestWindows = windowScores.slice(0, Math.min(5, windowScores.length));
```

Replace it with:

```tsx
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
```

This is a pure refactor — no behavior change. `windowSize` is still used later in the loop that slices segments for MFCC (`resampled.slice(win.start, win.start + windowSize)`); since `windowSize` is no longer declared locally, add it back as a one-line const right before that loop. Using Edit, find:

```tsx
      const norm = normRef.current!;
      const segmentResults: { normal: number; abnormal: number; score: number }[] = [];

      for (const win of bestWindows) {
        const segment = resampled.slice(win.start, win.start + windowSize);
```

Replace with:

```tsx
      const norm = normRef.current!;
      const windowSize = SAMPLE_RATE * SEGMENT_DURATION;
      const segmentResults: { normal: number; abnormal: number; score: number }[] = [];

      for (const win of bestWindows) {
        const segment = resampled.slice(win.start, win.start + windowSize);
```

- [ ] **Step 3: Build to verify no TypeScript errors**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully` and `Finished TypeScript` with no errors.

- [ ] **Step 4: Write the regression test**

Create `verify/window_score.mjs`:

```js
// Regression test for the window-quality scorer extracted from processSound
// in Task 1. Standalone port (same technique as verify/mfcc.mjs) so it can
// run in plain Node without a browser or React.

function scoreAudioWindow(segment) {
  let rms = 0;
  for (let i = 0; i < segment.length; i++) rms += segment[i] * segment[i];
  rms = Math.sqrt(rms / segment.length);
  if (rms < 0.0005) return { score: 0, rms };

  const chunkSize = Math.floor(segment.length / 10);
  const chunkEnergies = [];
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

function makeClip(n, fn) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = fn(i);
  return out;
}

// seeded PRNG so the test is deterministic (no Math.random())
function makeRand(seed) {
  let s = seed;
  return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
}

const SR = 2000;
const N = SR * 3; // 3-second window, matches SEGMENT_DURATION

let allPass = true;
function check(label, score, wantPass) {
  const got = score > 0.5;
  const ok = got === wantPass;
  allPass = allPass && ok;
  console.log(`${label.padEnd(40)} score=${score.toFixed(3)} pass=${got} want=${wantPass} ${ok ? "OK" : "FAIL"}`);
}

// A: near-silence -> score must be exactly 0 (below the 0.0005 rms floor)
const silence = makeClip(N, () => 0.0001);
check("near-silence", scoreAudioWindow(silence).score, false);

// B: steady moderate tone -> high rms score, ~zero cv -> should PASS
const steadyTone = makeClip(N, i => 0.05 * Math.sin((2 * Math.PI * 40 * i) / SR));
check("steady moderate tone", scoreAudioWindow(steadyTone).score, true);

// C: erratic bursty noise -> high cv across chunks -> should FAIL
const rand = makeRand(7);
const burstyNoise = makeClip(N, i => {
  const chunk = Math.floor(i / (N / 10));
  const chunkAmp = chunk % 3 === 0 ? 0.3 : 0.01; // wildly uneven energy per chunk
  return chunkAmp * (rand() * 2 - 1);
});
check("erratic bursty noise", scoreAudioWindow(burstyNoise).score, false);

// D: clipping/too-loud steady tone -> rms > 0.5 caps rmsScore at 0.3 -> should FAIL
const loudTone = makeClip(N, i => 0.9 * Math.sin((2 * Math.PI * 40 * i) / SR));
check("clipping steady tone", scoreAudioWindow(loudTone).score, false);

console.log(allPass ? "\nRESULT: PASS ✓" : "\nRESULT: FAIL ✗");
process.exit(allPass ? 0 : 1);
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd "C:/Users/vinay/projects/cardio/cardiolisten" && node verify/window_score.mjs`
Expected: all four `OK` lines, `RESULT: PASS ✓`, exit code 0.

- [ ] **Step 6: Re-run the existing MFCC and model regression tests to confirm the refactor didn't break anything**

Run: `node verify/compare.mjs` — expected: `RESULT: PASS ✓ (MFCC matches librosa within tolerance)`, max error `0.0000`.

- [ ] **Step 7: Commit**

```bash
git add pwa/app/page.tsx verify/window_score.mjs
git commit -m "refactor: extract reusable window quality scorer, add regression test"
```

---

### Task 2: Live beat-envelope detector (pure functions), regression test

**Files:**
- Modify: `pwa/app/page.tsx` (insert after the new "Heart sound window quality scoring" section added in Task 1, i.e. immediately before `// ============ PPG Signal Processing ============`)
- Create: `verify/beat_envelope.mjs`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `interface BeatEnvelopeState { fastEnv: number; slowEnv: number; lastBeatMs: number }`, `function initBeatEnvelopeState(): BeatEnvelopeState`, `function processBeatEnvelopeSample(state: BeatEnvelopeState, sampleAbs: number, nowMs: number): { state: BeatEnvelopeState; beatDetected: boolean }`. Task 4 consumes all three, calling `processBeatEnvelopeSample` once per animation frame (~60Hz) — the tuning constants below assume that call rate, not raw audio sample rate.

- [ ] **Step 1: Add the pure detector functions to `pwa/app/page.tsx`**

Using Edit, find (the end of the block Task 1 just added):

```tsx
// Best (highest-score) window found in `clip`; { score: 0, rms: 0 } if none.
function bestWindowScore(clip: Float32Array, sampleRate: number, windowSeconds: number, hopSeconds: number): WindowScore {
  const { windows } = scoreAudioWindows(clip, sampleRate, windowSeconds, hopSeconds);
  let best: WindowScore = { score: 0, rms: 0 };
  for (const w of windows) if (w.score > best.score) best = w;
  return best;
}

// ============ PPG Signal Processing ============
```

Replace with:

```tsx
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
```

- [ ] **Step 2: Build to verify no TypeScript errors**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully`, no errors. (These functions aren't called from anywhere yet — Task 4 wires them up — so this only checks syntax/types.)

- [ ] **Step 3: Write the regression test**

Create `verify/beat_envelope.mjs`:

```js
// Regression test for the live beat-envelope detector (Task 2). Standalone
// port of the pure state-transition function. Simulates a ~60Hz call rate
// (one call per virtual "animation frame") matching how Task 4 wires it up.

function initBeatEnvelopeState() {
  return { fastEnv: 0, slowEnv: 0, lastBeatMs: -Infinity };
}
function processBeatEnvelopeSample(state, sampleAbs, nowMs) {
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

function makeRand(seed) {
  let s = seed;
  return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
}

const FRAME_MS = 1000 / 60; // ~16.67ms, matches requestAnimationFrame cadence
const rand = makeRand(99);

let state = initBeatEnvelopeState();
let t = 0;
let beatsInSilence = 0;
let beatsInPulseTrain = 0;
let beatsInFlatNoise = 0;

// Phase 1: 2s of near-silence
for (; t < 2000; t += FRAME_MS) {
  const sample = 0.001 + rand() * 0.0005;
  const r = processBeatEnvelopeSample(state, sample, t);
  state = r.state;
  if (r.beatDetected) beatsInSilence++;
}

// Phase 2: 8 cardiac cycles at 75 BPM (800ms period), each with two brief
// bursts (S1 "lub" then S2 "dub" ~150ms later), amid quiet background.
const cycleMs = 800;
const pulseStart = t;
for (let cycle = 0; cycle < 8; cycle++) {
  const cycleBase = pulseStart + cycle * cycleMs;
  for (; t < cycleBase + cycleMs; t += FRAME_MS) {
    const intoCycle = t - cycleBase;
    const isS1 = intoCycle >= 0 && intoCycle < 40;
    const isS2 = intoCycle >= 150 && intoCycle < 190;
    const sample = (isS1 || isS2) ? 0.3 + rand() * 0.05 : 0.002 + rand() * 0.001;
    const r = processBeatEnvelopeSample(state, sample, t);
    state = r.state;
    if (r.beatDetected) beatsInPulseTrain++;
  }
}

// Phase 3: 2s of flat, low-level noise jitter with no prominent bursts
const flatStart = t;
for (; t < flatStart + 2000; t += FRAME_MS) {
  const sample = 0.01 + rand() * 0.02;
  const r = processBeatEnvelopeSample(state, sample, t);
  state = r.state;
  if (r.beatDetected) beatsInFlatNoise++;
}

console.log(`silence phase: ${beatsInSilence} beats (want 0)`);
console.log(`pulse-train phase: ${beatsInPulseTrain} beats (want 10-18, i.e. ~2 per cycle x 8 cycles)`);
console.log(`flat-noise phase: ${beatsInFlatNoise} beats (want <= 2, i.e. essentially none)`);

const pass = beatsInSilence === 0 && beatsInPulseTrain >= 10 && beatsInPulseTrain <= 18 && beatsInFlatNoise <= 2;
console.log(pass ? "\nRESULT: PASS ✓" : "\nRESULT: FAIL ✗");
process.exit(pass ? 0 : 1);
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "C:/Users/vinay/projects/cardio/cardiolisten" && node verify/beat_envelope.mjs`
Expected: `silence phase: 0 beats`, `pulse-train phase:` a number between 10 and 18, `flat-noise phase:` 0 or 1 or 2, `RESULT: PASS ✓`, exit code 0.

If the pulse-train count is outside range, adjust `fastAlpha`/`slowAlpha`/the `1.8x` threshold multiplier in **both** `pwa/app/page.tsx` and this test file together (they must stay identical — this is a port, not an independent implementation) and re-run.

- [ ] **Step 5: Commit**

```bash
git add pwa/app/page.tsx verify/beat_envelope.mjs
git commit -m "feat: add live beat-envelope detector for ECG-style waveform"
```

---

### Task 3: Guided-flow state machine (types, state, refs, callbacks) — no UI yet

**Files:**
- Modify: `pwa/app/page.tsx` (type additions near line 24; state additions near lines 511–516; new refs near the existing sound refs; new callbacks replacing `startSound`/`stopSound`/`processSound`, currently lines 788–987)

**Interfaces:**
- Consumes: `scoreAudioWindows`, `bestWindowScore` (Task 1), `initBeatEnvelopeState`, `processBeatEnvelopeSample`, `BeatEnvelopeState` (Task 2).
- Produces: `type ChestZone = "lower_left" | "upper_left" | "center"`, `const ZONE_ORDER: ChestZone[]`, `const ZONE_INFO: Record<ChestZone, { title: string; instruction: string }>`, callbacks `startGuidedFlow(): Promise<void>`, `beginZoneCheck(): void`, `stopGuidedFlow(): void`, `useBestZoneAnyway(): void`, `retryGuidedFlow(): void`, state `zoneIndex: number` (read via `ZONE_ORDER[zoneIndex]` for the current zone), `zoneMessage: string | null`, `checkCountdown: number`, `soundCountdown: number` (repurposed), `soundWaveform: number[]` (repurposed to carry 0/1 spike values instead of raw samples). Task 4 consumes all of the above to build the render blocks.

- [ ] **Step 1: Add the `ChestZone` type and zone metadata**

Using Edit, find (currently line 24):

```tsx
type SoundState = "idle" | "recording" | "processing" | "done";
```

Replace with:

```tsx
type SoundState = "idle" | "positioning" | "checking" | "recording" | "processing" | "done" | "all_zones_failed";
type ChestZone = "lower_left" | "upper_left" | "center";
const ZONE_ORDER: ChestZone[] = ["lower_left", "upper_left", "center"];
const ZONE_INFO: Record<ChestZone, { title: string; instruction: string }> = {
  lower_left: { title: "Lower-left chest", instruction: "Place the mic on your lower-left chest, near your ribs. This spot is usually clearest." },
  upper_left: { title: "Upper-left chest", instruction: "Place the mic on your upper-left chest, just below your collarbone." },
  center: { title: "Center chest", instruction: "Place the mic on the center of your chest, over your lower breastbone." },
};
```

- [ ] **Step 2: Add the new timing/threshold constants**

Using Edit, find (currently line 10):

```tsx
const RECORD_DURATION = 30;    // seconds of chest recording
```

Replace with:

```tsx
const CHECK_DURATION = 10;     // seconds to check each chest zone
const EXTEND_DURATION = 15;    // additional seconds recorded after a zone passes (25s total at that zone)
const GATE_PASS_THRESHOLD = 0.5; // scoreAudioWindow threshold for "clean enough to use" (see design spec)
```

- [ ] **Step 3: Update the sound state block and add guided-flow state/refs**

Using Edit, find (currently lines 511–516):

```tsx
  // Sound state
  const [soundState, setSoundState] = useState<SoundState>("idle");
  const [soundResult, setSoundResult] = useState<SoundResult | null>(null);
  const [soundCountdown, setSoundCountdown] = useState(RECORD_DURATION);
  const [soundWaveform, setSoundWaveform] = useState<number[]>([]);
  const [soundDiag, setSoundDiag] = useState<SoundDiag | null>(null);
  const [soundDebug, setSoundDebug] = useState<string | null>(null);
```

Replace with:

```tsx
  // Sound state
  const [soundState, setSoundState] = useState<SoundState>("idle");
  const [soundResult, setSoundResult] = useState<SoundResult | null>(null);
  const [soundCountdown, setSoundCountdown] = useState(EXTEND_DURATION);
  const [checkCountdown, setCheckCountdown] = useState(CHECK_DURATION);
  const [soundWaveform, setSoundWaveform] = useState<number[]>([]);
  const [soundDiag, setSoundDiag] = useState<SoundDiag | null>(null);
  const [soundDebug, setSoundDebug] = useState<string | null>(null);
  const [zoneMessage, setZoneMessage] = useState<string | null>(null);
  // Mirrors zoneIndexRef (see torchOn/pulseDiag mirroring above) so
  // useCallback([]) closures read the live zone, not the first-render one.
  const [zoneIndex, setZoneIndexState] = useState(0);
  const zoneIndexRef = useRef(0);
  const setZoneIndex = useCallback((v: number) => { zoneIndexRef.current = v; setZoneIndexState(v); }, []);
```

- [ ] **Step 4: Add guided-flow refs**

Using Edit, find the existing sound refs block:

```tsx
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const soundTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const soundAnimRef = useRef<number>(0);
  const soundCtxRef = useRef<AudioContext | null>(null);
  const soundLevelPeakRef = useRef(0);
```

Replace with:

```tsx
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
```

- [ ] **Step 5: Replace `startSound`/`stopSound`/`processSound` with the guided-flow callbacks**

This block was already touched once by Task 1 (the `processSound` scoring loop refactor), so the exact current text differs from the very first read of this file — the block below is the exact **post-Task-1** text (Task 2 doesn't touch this block at all). Using Edit, find this exact block:

```tsx
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
```

Replace the whole block with:

```tsx
  // ============ SOUND LOGIC (guided multi-zone flow) ============

  const teardownGuidedStream = useCallback(() => {
    if (audioStreamRef.current) { audioStreamRef.current.getTracks().forEach(t => t.stop()); audioStreamRef.current = null; }
    if (soundCtxRef.current) { soundCtxRef.current.close(); soundCtxRef.current = null; }
    analyserRef.current = null;
  }, []);

  const startGuidedFlow = useCallback(async () => {
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
    if (soundTimerRef.current) clearInterval(soundTimerRef.current);
    cancelAnimationFrame(soundAnimRef.current);
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") { recorder.onstop = null; recorder.stop(); }
    teardownGuidedStream();
    setSoundState("idle");
    setZoneMessage(null);
  }, []);

  const analyzeZoneAudio = useCallback(async (chunks: Blob[], zone: ChestZone) => {
    setSoundState("processing");
    try {
      const blob = new Blob(chunks, { type: "audio/webm" });
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      audioCtx.close();

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
          recorder.onstop = () => { analyzeZoneAudio(audioChunksRef.current.slice(), zone); };
          recorder.stop();
        }
      }
    }, 1000);
  }, []);

  const advanceZone = useCallback(() => {
    cancelAnimationFrame(soundAnimRef.current);
    const recorder = mediaRecorderRef.current;
    const nextIndex = zoneIndexRef.current + 1;

    const goNext = () => {
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
    const zone = ZONE_ORDER[zoneIndexRef.current];
    const chunksSoFar = audioChunksRef.current.slice();
    const blob = new Blob(chunksSoFar, { type: "audio/webm" });

    let score = 0;
    try {
      const arrayBuffer = await blob.arrayBuffer();
      const scratchCtx = new AudioContext();
      const audioBuffer = await scratchCtx.decodeAudioData(arrayBuffer);
      scratchCtx.close();
      const resampled = resampleAudio(audioBuffer.getChannelData(0), audioBuffer.sampleRate, SAMPLE_RATE);
      score = bestWindowScore(resampled, SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_DURATION / 2).score;
    } catch {
      score = 0;
    }

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
    analyzeZoneAudio(chunks, bestZone);
  }, []);

  const retryGuidedFlow = useCallback(() => {
    zoneChunksRef.current = {};
    zoneScoresRef.current = {};
    setZoneIndex(0);
    setZoneMessage(null);
    setSoundState("positioning");
  }, []);
```

- [ ] **Step 6: Build to verify no TypeScript errors**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully` and `Finished TypeScript` with no errors. There will likely be "declared but never used" warnings/errors for `startGuidedFlow`, `beginZoneCheck`, `stopGuidedFlow`, `useBestZoneAnyway`, `retryGuidedFlow`, `checkCountdown`, `zoneMessage`, `zoneIndex` since the render block (Task 4) doesn't reference them yet. If the build fails on unused-variable errors (rather than just warnings), that's expected until Task 4 — confirm the *only* errors are unused-declaration errors for exactly those names, not anything else, then proceed; Task 4 resolves all of them.

- [ ] **Step 7: Commit**

```bash
git add pwa/app/page.tsx
git commit -m "feat: add guided multi-zone sound capture state machine (no UI yet)"
```

---

### Task 4: Zone map + checking/recording UI, ECG-style waveform wiring

**Files:**
- Modify: `pwa/app/page.tsx` (the Sound tab render block, currently lines 1150–1254; the idle-state screen and "recording" screen within it)

**Interfaces:**
- Consumes: everything from Task 3 (`ZONE_ORDER`, `ZONE_INFO`, `ChestZone`, `startGuidedFlow`, `beginZoneCheck`, `stopGuidedFlow`, `soundState`, `zoneIndex`, `zoneMessage`, `checkCountdown`, `soundCountdown`, `soundWaveform`, `soundDiag`). Reuses the existing `LiveWaveform` component unmodified — the ECG look comes entirely from the 0/1 spike data now being pushed into `soundWaveform` by `beginZoneCheck` (Task 3), not from a new rendering component.
- Produces: a `ZoneMap` component, `{ activeZone: ChestZone }` props, rendered inline in this task (not exported/reused elsewhere).

- [ ] **Step 1: Add the `ZoneMap` component**

Using Edit, find (right before the main component, currently around line 481):

```tsx
// ============ MAIN COMPONENT ============
```

Replace with:

```tsx
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
```

- [ ] **Step 2: Replace the Sound tab idle screen**

Using Edit, find (currently lines 1153–1171):

```tsx
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
```

Replace with:

```tsx
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
```

- [ ] **Step 3: Update the "recording" (extended capture) screen**

Using Edit, find (currently lines 1173–1209, now directly following the block Step 2 just added):

```tsx
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
```

Replace with:

```tsx
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
```

- [ ] **Step 4: Update "Record Again" to reset guided-flow state**

Using Edit, find (near the end of the "done" block):

```tsx
                <button onClick={() => { setSoundState("idle"); setSoundResult(null); setSoundWaveform([]); }} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Record Again
                </button>
```

Replace with:

```tsx
                <button onClick={() => { setSoundState("idle"); setSoundResult(null); setSoundWaveform([]); setZoneIndex(0); zoneChunksRef.current = {}; zoneScoresRef.current = {}; }} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-sm font-medium transition-colors">
                  Record Again
                </button>
```

- [ ] **Step 5: Build to verify no TypeScript errors**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully`, no errors. The only remaining unused-declaration errors, if any, should be `useBestZoneAnyway` and `retryGuidedFlow` (Task 5 wires those up) — confirm no other errors.

- [ ] **Step 6: Commit**

```bash
git add pwa/app/page.tsx
git commit -m "feat: add zone map, checking screen, and ECG-style live waveform UI"
```

---

### Task 5: "All zones failed" screen

**Files:**
- Modify: `pwa/app/page.tsx` (Sound tab render block, insert a new conditional block alongside the others added in Task 4)

**Interfaces:**
- Consumes: `useBestZoneAnyway`, `retryGuidedFlow` (Task 3).

- [ ] **Step 1: Add the `all_zones_failed` render block**

Using Edit, find the `{soundState === "processing" && (` block (immediately after the "recording" block from Task 4):

```tsx
            {soundState === "processing" && (
              <div className="flex flex-col items-center gap-4 py-8">
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <p className="text-blue-400 text-sm">Analyzing best segments...</p>
                <p className="text-slate-500 text-xs">Selecting the cleanest windows for inference</p>
              </div>
            )}
```

Replace with:

```tsx
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
```

- [ ] **Step 2: Build to verify no TypeScript errors**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully`, `Finished TypeScript`, zero errors, zero unused-declaration warnings related to this feature.

- [ ] **Step 3: Commit**

```bash
git add pwa/app/page.tsx
git commit -m "feat: add all-zones-failed screen with use-best/try-again options"
```

---

### Task 6: Extend diagnostics logging with zone context

**Files:**
- Modify: `pwa/app/page.tsx` (already-added `pushDiagLog` calls in Task 3 include `zone=` — this task verifies and, if needed, extends the "Copy diagnostics" summary text)

**Interfaces:**
- Consumes: `pushDiagLog`, `readDiagLog` (existing, unchanged).

- [ ] **Step 1: Verify zone context is present in all sound diagnostic log lines**

Using Grep (or manual read), confirm every `pushDiagLog(` call added in Task 3 for the sound flow includes `zone=${zone}` or equivalent zone identification:
- `evaluateZoneGate`: `SOUND ZONE CHECK: zone=${zone} score=...`
- `analyzeZoneAudio` success: `SOUND OK: zone=${zone} label=...`
- `analyzeZoneAudio` failure (no usable windows): the `debugBase` string already includes `zone=${zone}` at its start (added in Task 3's `analyzeZoneAudio`), and it's interpolated into the `SOUND FAIL:` log line.
- `analyzeZoneAudio` catch block: `SOUND ERROR: zone=${zone} ...`

No code change expected here if Task 3 was implemented as specified — this step is a verification checkpoint, not new code. If any call is missing zone context, add it following the pattern of the others.

- [ ] **Step 2: Manual verification of the copy-diagnostics output**

This requires a browser (not just `npm run build`) since `localStorage`/`navigator.clipboard` don't exist in Node:

Run: `cd pwa && npm run dev`, open `http://localhost:3000` in a browser, go to the Heart Sound tab, click "Start Guided Check", allow mic access, then click "Cancel" after a few seconds (don't need to complete a full check for this verification — just need at least one zone-check attempt logged). If a `SOUND ZONE CHECK` line was logged before cancelling, the "Copy diagnostics" button should appear under the tab bar.

Expected: clicking "Copy diagnostics" and pasting the clipboard contents shows a line containing `SOUND ZONE CHECK: zone=lower_left score=... PASS` or `FAIL`.

- [ ] **Step 3: Commit (only if Step 1 required a code change)**

```bash
git add pwa/app/page.tsx
git commit -m "fix: ensure zone context present in all sound diagnostic log lines"
```

If Step 1 found no gaps, skip this commit — there's nothing to commit.

---

### Task 7: Full regression pass, manual verification, deploy

**Files:** None (verification and deployment only).

- [ ] **Step 1: Run every regression test**

Run:
```bash
cd "C:/Users/vinay/projects/cardio/cardiolisten"
node verify/window_score.mjs
node verify/beat_envelope.mjs
node verify/compare.mjs
node verify/ppg.mjs
```

Expected: all four print `RESULT: PASS ✓` with exit code 0. `verify/compare.mjs` must still show `max abs err: 0.0000` (confirms the Task 1 refactor didn't touch the MFCC path). `verify/ppg.mjs` confirms the Pulse tab (untouched by this feature) still passes.

- [ ] **Step 2: Run the end-to-end model check**

Run (from `pwa/`, per the pattern established earlier in this project — copy `verify/model_e2e.mjs` in, fix its relative import, run, then remove the copy):

```bash
cd "C:/Users/vinay/projects/cardio/cardiolisten"
cp verify/model_e2e.mjs pwa/_e2e_tmp.mjs
sed -i 's#"./mfcc.mjs"#"../verify/mfcc.mjs"#; s#"pwa/public/model"#"public/model"#' pwa/_e2e_tmp.mjs
cd pwa && node _e2e_tmp.mjs 2>&1 | tail -8
rm -f _e2e_tmp.mjs
```

Expected: `softmax valid ✓`, `model responds to input (not degenerate) ✓`.

- [ ] **Step 3: Production build**

Run: `cd pwa && npm run build`
Expected: `✓ Compiled successfully`, `Finished TypeScript` with zero errors, static pages generated.

- [ ] **Step 4: Manual smoke test (requires a browser with mic access — real device preferred, since this feature's whole purpose is real-hardware reliability)**

Run: `cd pwa && npm run dev`, open in a browser (or deploy to a preview URL first if local mic testing isn't practical), then:
1. Go to Heart Sound tab, click "Start Guided Check" — should prompt for mic permission once.
2. Zone map should appear with "Lower-left chest" highlighted first (not upper-left, despite it being visually top-left in some layouts — confirms attempt order is decoupled from visual position per the spec).
3. Tap "I'm in position" — should show a 10-second countdown with a flat scrolling trace that spikes when you tap/scratch near the mic (simulates a "beat").
4. Let it run past 10s with no signal (e.g., mic covered but silent) — should show "Didn't get a clean signal here" and advance to the next zone.
5. Provide a clear rhythmic tap pattern (or real chest audio) at a zone — should show "Detected! Recording a bit more..." and transition to the 15s extended capture, then to "Analyzing..." and a result screen.
6. Verify "Copy diagnostics" produces a paste-able history showing the zone attempts.

Document the actual observed behavior (pass/fail per point above) before proceeding — this plan's Task 7 is not complete until this is done on at least one real device, consistent with how the Pulse tab fixes earlier in this project required real Pixel 6 Pro verification, not just simulated tests.

- [ ] **Step 5: Deploy**

Run:
```bash
cd pwa && vercel deploy --prod --yes --token "$VERCEL_TOKEN_2"
```

Expected: `Aliased: https://pwa-ten-tawny.vercel.app` (same URL as previous deploys — see Vercel account rules in the project's global CLAUDE.md: this project is on the newer account, `vinaysolapurkar22-9169`, because the older default account hit its 200-project Hobby limit).

- [ ] **Step 6: Verify the deployment is live**

Run: `curl -s -o /dev/null -w "live: %{http_code}\n" "https://pwa-ten-tawny.vercel.app/"`
Expected: `live: 200`.

- [ ] **Step 7: Final commit (if any fixes were made during manual verification)**

```bash
git add -A
git commit -m "fix: address issues found during guided-auscultation manual verification"
```

If no fixes were needed, skip — nothing to commit.
