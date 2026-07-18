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
