"""Generate a deterministic test signal + reference librosa MFCC to validate
the in-browser MFCC reimplementation against the training pipeline."""
import json
import numpy as np
import librosa

SR = 2000
DUR = 3
N_MFCC = 20
HOP = 128

rng = np.random.default_rng(42)
t = np.arange(SR * DUR) / SR
# Synthetic heart-sound-ish signal: S1/S2 thumps ~1 Hz + harmonics + noise
sig = np.zeros_like(t)
for beat in np.arange(0, DUR, 0.85):
    for (off, freq, amp) in [(0.0, 45, 1.0), (0.32, 60, 0.7)]:
        c = beat + off
        env = np.exp(-((t - c) ** 2) / (2 * 0.02 ** 2))
        sig += amp * env * np.sin(2 * np.pi * freq * (t - c))
sig += 0.02 * rng.standard_normal(t.shape)
sig = sig.astype(np.float32)

mfcc = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)  # (20, frames)
print("signal len:", len(sig), "mfcc shape:", mfcc.shape, "librosa", librosa.__version__)

with open("verify/ref.json", "w") as f:
    json.dump({
        "signal": sig.tolist(),
        "mfcc": mfcc.T.tolist(),  # (frames, 20) to match JS layout
        "librosa": librosa.__version__,
    }, f)
print("wrote verify/ref.json")
