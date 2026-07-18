# Guided Chest Placement + Live ECG-Style Waveform (Heart Sound tab)

## Problem

The current Heart Sound flow asks the user to press the phone mic somewhere on
their chest and blindly record for 30 seconds. Placement quality is the
single biggest lever on recording quality, and the app currently does nothing
to help find a good spot — it just hopes the whole 30s was recorded from a
usable position. The live waveform during recording is also raw, high-frequency
mic samples, which looks like random static ("gibberish up and down") rather
than something that visibly tracks heartbeats.

## Scope

Heart Sound tab only. The Pulse (fingertip camera) tab and the already-verified
MFCC/model inference pipeline (`extractMfcc`, `resampleAudio`, the librosa-exact
feature extraction, the TF.js model call) are unchanged. This is purely a new
capture-UX layer in front of the existing, tested inference code.

## User-approved decisions

- **3-zone simplified map** (not the full 4-point clinical Aortic/Pulmonic/
  Tricuspid/Mitral map) — plain language, no clinical jargon, since the
  audience isn't clinicians.
- **Tap "I'm in position" to advance** — not fully automatic — to avoid
  mistaking hand-repositioning rustle for real signal.
- **Visual style**: abstract rounded-rectangle zone grid (not a body
  silhouette) with 3 numbered/labeled zones.
- **Live waveform**: ECG-style flat scrolling trace with a sharp spike at each
  detected beat (matches what the user explicitly asked for), rendered in the
  app's existing blue (`#3b82f6`) sound-tab accent.

## Flow

1. **Zone map screen** (new `SoundState` value, e.g. `"positioning"`): shows
   the 3-zone grid (fixed visual layout: top-left, bottom-left, bottom-right
   rounded rectangles, matching the approved mockup). The grid's on-screen
   positions never change, but the **attempt order** is independent of that
   layout — the currently-active zone is highlighted on the map (border/glow)
   regardless of where it sits visually. Attempt order:
   1. Lower-left chest (near your ribs) — labeled "usually clearest," **tried
      first**, since that's the anatomically loudest spot (apex/mitral area)
      and minimizes expected time-to-success.
   2. Upper-left chest (below your collarbone)
   3. Center chest (lower breastbone)
   Each zone has a short plain-language placement instruction and a
   "I'm in position" button. No numbers are shown to the user (avoids
   implying zone 1 on the map must be tried first) — only the highlight and
   the instruction text identify the current zone.

2. **Mic opens once**, at the start of the whole guided flow (first tap of
   "I'm in position" for zone 1), and stays open across all zones/re-tries —
   no repeated `getUserMedia` permission prompts or AudioContext re-init
   between zones. Implementation: keep the existing raw-audio capture setup
   (no AGC/NS/EC, matches training pipeline) but don't tear down the stream
   between zones; only stop it once a final result is reached or the user
   cancels.

3. **Per-zone check** (new `SoundState` value `"checking"`), 10 seconds:
   - Live ECG-style trace renders in real time (see "Live waveform" below).
   - In parallel, a rolling quality score is computed (see "Pass/fail gate"
     below) over the captured samples for that zone.
   - At the 10s mark, evaluate the gate:
     - **Pass** → show "Detected! Hold on, calculating..." → transition to
       extended capture at the *same* zone (see step 4).
     - **Fail** → show "Didn't get a clean signal here" for ~1.5s → advance
       to the next zone's "I'm in position" screen automatically. If this was
       the 3rd (last) zone, go to step 5 (all zones failed) instead.

4. **Extended capture on success** (reuses existing `SoundState = "recording"`
   UI/countdown, relabeled): continue recording at the confirmed-good zone for
   15 more seconds, so the *total* usable audio at that zone is ~25s — close
   to today's 30s `RECORD_DURATION`, enough for the existing best-5-of-N
   3-second-window selection in `processSound` to have a comparable number of
   candidate windows (~9 at 50% hop over 25s vs. ~19 over 30s today; still
   comfortably above the 5 it keeps). Only the audio from *this* zone is used
   for inference — the failed zones' audio (if any) is discarded, not mixed
   in.

5. **All 3 zones failed the gate**: show two options —
   - **"Use best zone anyway"** — proceed to extended capture + inference
     using whichever of the 3 zones scored highest, with a visible low-quality
     warning on the eventual result (reusing the existing `quality < 0.25`
     warning path already in the result screen).
   - **"Try again"** — restart the whole guided flow from zone 1, mic stream
     kept open.

6. **Processing / done**: unchanged from today — same `extractMfcc` →
   normalize → `tf.tensor3d` → `model.predict` → weighted-average pipeline,
   same result screen, same diagnostics logging (`pushDiagLog`) — extended to
   note which zone succeeded and how many zones were tried.

## Pass/fail gate (per zone, at the 10s mark)

Reuse the existing, already-verified window quality heuristic from
`processSound` — RMS in a sane range (not near-silence, not clipping) combined
with low coefficient-of-variation of chunk energies (a proxy for "consistent,
rhythmic energy" vs. noise) — rather than building a new detection algorithm.
Concretely: run the same scoring computation the current code already applies
to candidate 3-second windows (`score = rmsScore * cvScore`, where
`rmsScore` is `1` if `0.001 < rms < 0.5` else `0.3`, and
`cvScore = max(0, 1 - cv * 2)`), applied here to 3-second sub-windows within
the zone's 10s check clip (at 50% hop, giving ~5 sub-windows); take the best
sub-window's score.

**Pass threshold: `score > 0.5`.** This requires `rmsScore == 1` (RMS in the
non-silent, non-clipping range) *and* `cvScore > 0.5` (`cv < 0.25`, i.e.
energy across the 10 sub-chunks of that window is reasonably consistent —
more rhythmic than noisy). This is a concrete, testable number (see Testing
below), not a relative "best of the 3 zones" comparison — a zone can fail
outright even if it's the best of a bad set, which is what drives the
"all 3 zones failed" path.

This is a deliberate low-risk choice: no new algorithm surface, reuses logic
that's already running in production today.

## Live waveform (ECG-style)

A **separate, lightweight, purely-visual** mechanism from the pass/fail gate
above — it does not feed the model or the gate decision, so it doesn't need to
be clinically precise, only responsive and honest (no fake/randomized spikes).

Implementation: a real-time envelope follower over the live analyser data
already being read each frame (rectify + short moving-average smoothing),
feeding an adaptive-threshold peak detector (fires when short-term energy
exceeds roughly 1.8x the recent rolling average) with a short refractory
period (~150ms) — short enough to allow two close spikes per cardiac cycle
(S1 "lub" + S2 "dub"), which is what a real heart-sound trace looks like,
rather than collapsing each heartbeat into one spike.

Rendering: flat scrolling baseline (matches the ECG mockup approved in
brainstorming) with a sharp spike drawn at each detected peak event, using the
existing `LiveWaveform`-style SVG polyline approach already in the codebase,
replacing the raw-sample feed with the envelope+spike feed for this screen
only. The existing raw-waveform component/behavior is untouched elsewhere if
still needed.

## State machine changes

Current: `SoundState = "idle" | "recording" | "processing" | "done"`

New: `SoundState = "idle" | "positioning" | "checking" | "recording" |
"processing" | "done" | "all_zones_failed"`

- `idle`: existing start screen, now says "we'll help you find the clearest
  spot" instead of "press mic on chest."
- `positioning`: zone map + current zone's placement instruction + "I'm in
  position" button.
- `checking`: 10s per-zone check, ECG-style live trace, gate evaluation at the
  end.
- `recording`: extended 15s capture at a confirmed-good zone (relabeled
  version of today's recording screen).
- `processing`/`done`: unchanged.
- `all_zones_failed`: the "use best anyway / try again" choice screen.

## Diagnostics

Extend the existing `pushDiagLog` sound entries to include: which zone
succeeded (or that all 3 failed and "use best anyway" was chosen), each
zone's gate score, and total zones attempted — so the existing "Copy
diagnostics" flow still gives a complete picture of what happened, now
including the placement search.

## Testing

- Extend `verify/` with a test that feeds the existing window-quality scoring
  function synthetic "good" (periodic, moderate-RMS) and "bad" (silence,
  clipping, pure noise) 10s clips and asserts the gate passes/fails as
  expected — this is the one new piece of decision logic (an existing
  function applied to a new-length input), so it's worth a regression test
  before wiring it into the UI.
- Manual verification on real hardware (per the ongoing Pixel 6 Pro testing)
  is still required for the live spike detector's real-world feel, same as
  the pulse-detection changes earlier in this project.

## Out of scope (explicitly not doing)

- No change to the Pulse/camera tab.
- No change to the MFCC/model inference pipeline itself.
- No full 4-point clinical auscultation map (rejected in brainstorming in
  favor of the simpler 3-zone version).
- No body-silhouette illustration (rejected in favor of the abstract zone
  grid).
