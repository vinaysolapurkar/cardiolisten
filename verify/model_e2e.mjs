// End-to-end: load the real TF.js model in Node and run the new MFCC pipeline
// on distinct inputs to confirm it produces valid, discriminating softmax output.
import fs from "fs";
import path from "path";
import * as tf from "@tensorflow/tfjs";
import { extractMfcc } from "./mfcc.mjs";

const MODEL_DIR = "pwa/public/model";
const N_MFCC = 20, MAX_FRAMES = 47, SR = 2000, DUR = 3;

// Custom IOHandler to load a layers model from local disk (pure tfjs, no tfjs-node).
function fileIOHandler(dir) {
  return {
    load: async () => {
      const modelJSON = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
      const manifest = modelJSON.weightsManifest;
      const specs = [];
      const buffers = [];
      for (const group of manifest) {
        for (const p of group.paths) {
          buffers.push(fs.readFileSync(path.join(dir, p)));
        }
        specs.push(...group.weights);
      }
      const concat = Buffer.concat(buffers);
      const weightData = concat.buffer.slice(concat.byteOffset, concat.byteOffset + concat.byteLength);
      return {
        modelTopology: modelJSON.modelTopology,
        weightSpecs: specs,
        weightData,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy,
      };
    },
  };
}

function runPipeline(signal, norm, model) {
  const frames = extractMfcc(signal);
  const padded = [];
  for (let i = 0; i < MAX_FRAMES; i++) padded.push(i < frames.length ? frames[i] : new Float64Array(N_MFCC));
  const input = new Float32Array(MAX_FRAMES * N_MFCC);
  for (let i = 0; i < MAX_FRAMES; i++)
    for (let j = 0; j < N_MFCC; j++)
      input[i * N_MFCC + j] = (padded[i][j] - norm.mean[j]) / (norm.std[j] + 1e-8);
  const t = tf.tensor3d(input, [1, MAX_FRAMES, N_MFCC]);
  const out = model.predict(t);
  const probs = out.dataSync();
  t.dispose(); out.dispose();
  return [probs[0], probs[1]];
}

const main = async () => {
  const norm = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, "norm_params.json"), "utf8"));
  const model = await tf.loadLayersModel(fileIOHandler(MODEL_DIR));
  console.log("model loaded, input:", JSON.stringify(model.inputs[0].shape), "output:", JSON.stringify(model.outputs[0].shape));

  // Input A: rhythmic heartbeat-like signal
  const t = Array.from({ length: SR * DUR }, (_, i) => i / SR);
  const beat = new Float32Array(SR * DUR);
  for (let b = 0; b < DUR; b += 0.85)
    for (const [off, f, a] of [[0, 45, 1.0], [0.32, 60, 0.7]]) {
      for (let i = 0; i < beat.length; i++) {
        const env = Math.exp(-((t[i] - (b + off)) ** 2) / (2 * 0.02 ** 2));
        beat[i] += a * env * Math.sin(2 * Math.PI * f * (t[i] - (b + off)));
      }
    }
  // Input B: white noise
  let s = 7;
  const rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  const noise = new Float32Array(SR * DUR);
  for (let i = 0; i < noise.length; i++) noise[i] = rand() * 0.3;
  // Input C: near-silence
  const silence = new Float32Array(SR * DUR);
  for (let i = 0; i < silence.length; i++) silence[i] = rand() * 0.002;

  const A = runPipeline(beat, norm, model);
  const B = runPipeline(noise, norm, model);
  const C = runPipeline(silence, norm, model);
  console.log(`heartbeat-like -> normal=${A[0].toFixed(3)} abnormal=${A[1].toFixed(3)}`);
  console.log(`white-noise    -> normal=${B[0].toFixed(3)} abnormal=${B[1].toFixed(3)}`);
  console.log(`near-silence   -> normal=${C[0].toFixed(3)} abnormal=${C[1].toFixed(3)}`);

  const valid = [A, B, C].every(p => Math.abs(p[0] + p[1] - 1) < 1e-3 && p[0] >= 0 && p[1] >= 0);
  const discriminates = Math.abs(A[0] - B[0]) > 0.02 || Math.abs(A[0] - C[0]) > 0.02;
  console.log(valid ? "softmax valid ✓" : "softmax INVALID ✗");
  console.log(discriminates ? "model responds to input (not degenerate) ✓" : "model output is CONSTANT ✗");
  process.exit(valid && discriminates ? 0 : 1);
};
main().catch(e => { console.error(e); process.exit(1); });
