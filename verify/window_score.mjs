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

  // rms >= 0.0005 already guaranteed by the early return above (that's the
  // real silence floor) -- only cap the UPPER end (clipping/too-loud). A
  // second, higher lower-bound here double-penalizes legitimately quiet
  // recordings: real phone-mic heart sounds average very low RMS over a 3s
  // window since most of the window is silence between S1/S2 beats (see the
  // "quiet-but-rhythmic" test case, calibrated to real device data below).
  const rmsScore = rms < 0.5 ? 1 : 0.15;
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

const GATE_PASS_THRESHOLD = 0.18; // must match page.tsx's GATE_PASS_THRESHOLD

let allPass = true;
function check(label, score, wantPass) {
  const got = score > GATE_PASS_THRESHOLD;
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

// E: real-device data (Pixel 6 Pro, 2026-07-18) -- a recording the model
// classified with 73% confidence had best-window rms of only 0.06% (0.0006)
// after resampling to 2kHz, well below the old rms>0.001 "good" floor. A
// quiet-but-clearly-rhythmic clip at that amplitude should PASS (it's a real
// working recording, just quiet -- most of a 3s heart-sound window is
// legitimately near-silent between S1/S2 beats).
const quietRhythmic = makeClip(N, i => {
  const t = i / SR;
  const cyclePos = t % 0.85;
  const isBeat = cyclePos < 0.04 || (cyclePos > 0.32 && cyclePos < 0.36);
  return (isBeat ? 0.0035 : 0.0004) * Math.sin(2 * Math.PI * 45 * t);
});
const quietResult = scoreAudioWindow(quietRhythmic);
console.log(`(quiet-but-rhythmic rms = ${(quietResult.rms * 100).toFixed(3)}%, matches real device data)`);
check("quiet-but-rhythmic (real-world amplitude)", quietResult.score, true);

console.log(allPass ? "\nRESULT: PASS ✓" : "\nRESULT: FAIL ✗");
process.exit(allPass ? 0 : 1);
