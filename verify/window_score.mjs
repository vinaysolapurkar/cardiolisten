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
