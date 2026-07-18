import fs from "fs";
import { extractMfcc } from "./mfcc.mjs";

const ref = JSON.parse(fs.readFileSync("verify/ref.json", "utf8"));
const audio = Float32Array.from(ref.signal);
const jsFrames = extractMfcc(audio);
const pyFrames = ref.mfcc; // (frames, 20)

const nFrames = Math.min(jsFrames.length, pyFrames.length);
console.log(`frames: js=${jsFrames.length} py=${pyFrames.length} (librosa ${ref.librosa})`);

let maxAbs = 0, sumAbs = 0, cnt = 0;
let sumAbsPy = 0;
// per-coefficient error
const perCoefErr = new Array(20).fill(0);
const perCoefCnt = new Array(20).fill(0);
for (let t = 0; t < nFrames; t++) {
  for (let j = 0; j < 20; j++) {
    const a = jsFrames[t][j], b = pyFrames[t][j];
    const e = Math.abs(a - b);
    maxAbs = Math.max(maxAbs, e);
    sumAbs += e; sumAbsPy += Math.abs(b); cnt++;
    perCoefErr[j] += e; perCoefCnt[j]++;
  }
}
const meanAbs = sumAbs / cnt;
const meanMag = sumAbsPy / cnt;
console.log(`max abs err:  ${maxAbs.toFixed(4)}`);
console.log(`mean abs err: ${meanAbs.toFixed(4)}`);
console.log(`mean |librosa| magnitude: ${meanMag.toFixed(4)}`);
console.log(`relative mean err: ${(meanAbs / meanMag * 100).toFixed(2)}%`);
console.log("per-coef mean abs err (first 6):", perCoefErr.slice(0, 6).map((e, i) => (e / perCoefCnt[i]).toFixed(3)).join(", "));

// Sample: first frame first 5 coefs
console.log("js  frame0[0:5]:", Array.from(jsFrames[0].slice(0, 5)).map(x => x.toFixed(2)).join(", "));
console.log("py  frame0[0:5]:", pyFrames[0].slice(0, 5).map(x => x.toFixed(2)).join(", "));

const PASS = maxAbs < 1.0 && meanAbs < 0.2;
console.log(PASS ? "\nRESULT: PASS ✓ (MFCC matches librosa within tolerance)" : "\nRESULT: FAIL ✗ (MFCC diverges from librosa)");
process.exit(PASS ? 0 : 1);
