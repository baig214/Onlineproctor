# ProctorSense — Technical Documentation

> A fully client-side, real-time AI proctoring system that watches a webcam feed and flags suspicious exam behavior — with zero video ever leaving the browser.

---

## 1. What This Project Actually Is

ProctorSense is a browser-based exam/interview monitoring console. It runs an object-detection neural network, a facial gaze-tracking model, and a custom motion-detection pipeline **simultaneously, in real time, entirely inside the user's browser** — no server, no video upload, no backend inference. Everything from the YOLO object detector to the face-landmark model runs on-device via WebGPU/WASM.

It watches the candidate's webcam and continuously asks five questions:

| Signal | Question it answers |
|---|---|
| **Person presence** | Is the candidate still in frame? |
| **Multiple people** | Is someone else in the room? |
| **Phone** | Is there a phone visible — and is its screen on? |
| **Secondary device** | Is there a laptop/TV/tablet visible? |
| **Gaze** | Is the candidate looking at their own screen, or away/at a second screen? |
| **Background motion** | Did something move or change behind the candidate? |

Each of these signals feeds into a unified **risk engine** that accumulates evidence over time, decides when something is suspicious enough to raise a flag, captures visual proof, and renders it all into a live dashboard — risk meter, belief bars, incident timeline, history sidebar, and a full "advanced analytics" mode for reviewers.

The system is deliberately engineered to **not cry wolf**. Almost every detector here uses temporal smoothing, confirmation windows, and cooldowns specifically so that a single bad frame (motion blur, a hand passing in front of the camera, a brief misclassification) doesn't trigger a false incident. That tolerance for noise — while still catching real, sustained patterns — is the core design challenge the whole system is built around, and it shows up in almost every module below.

---

## 2. System Architecture — The Big Picture

The whole pipeline is built as a set of cooperating **Web Workers**, each with one job, talking to a single main thread that owns the UI.

```
                         ┌─────────────────────────┐
                         │   getUserMedia() webcam  │
                         └────────────┬─────────────┘
                                       │  MediaStreamTrackProcessor
                                       │  (raw VideoFrames, zero-copy)
                                       ▼
                    ┌──────────────────────────────────┐
                    │         Main Thread (UI)          │
                    │  - throttles to target FPS         │
                    │  - fans frames out to 3 workers    │
                    └──┬───────────────┬───────────────┬─┘
                       │               │               │
         every frame   │   ~10 FPS     │   every 5s    │
                       ▼               ▼               ▼
              ┌────────────────┐ ┌─────────────┐ ┌──────────────────┐
              │   worker.js     │ │gaze-worker.js│ │ motion-worker.js │
              │  YOLO26 object  │ │ MediaPipe    │ │ frame-diff       │
              │  detection      │ │ face/iris    │ │ background       │
              │  (ONNX/WebGPU)  │ │ tracking     │ │ motion detector  │
              └───────┬────────┘ └──────┬───────┘ └────────┬─────────┘
                      │                 │                   │
                      │  detections,    │  gaze state,      │  motion event +
                      │  person box,    │  away/oscillation │  before/after/diff
                      │  phone screen   │  metrics          │  bitmaps
                      │  state          │                   │
                      ▼                 ▼                   ▼
              ┌────────────────────────────────────────────────────┐
              │                  risk-engine.js                     │
              │   evidence accumulation · incident state machine    │
              │   correlation logic · risk score (0–100)            │
              └───────────────────────┬──────────────────────────────┘
                                       ▼
                      ┌──────────────────────────────┐
                      │     index.html (dashboard)    │
                      │  risk ring · belief bars ·     │
                      │  incident cards · timeline ·   │
                      │  history sidebar · analytics   │
                      └──────────────────────────────┘
```

Why three separate workers instead of one big loop? Because each task has a wildly different cost and required cadence:

- **Object detection** (`worker.js`) is the heaviest — a full neural net forward pass — so it runs at a *throttled, adaptive* frame rate.
- **Gaze tracking** (`gaze-worker.js`) needs to feel responsive (head/eye movement is fast), so it samples more frequently on a separate thread so it's never blocked waiting on the object detector.
- **Motion detection** (`motion-worker.js`) only needs occasional samples (it's looking for *sustained* background change, not split-second motion), so it intentionally runs the *least* often — once every 5 seconds — to save CPU.

Splitting them across workers means a slow YOLO inference never causes the gaze tracker to stutter, and vice versa — each pipeline lives on its own thread and communicates with the main thread purely via `postMessage`.

---

## 3. Object Detection Pipeline (`worker.js`)

### 3.1 The model

Detection runs on **YOLO26s**, exported to ONNX and executed in-browser via **onnxruntime-web** on the **WebGPU** backend — meaning real GPU-accelerated inference with no server round-trip. The model is loaded once when the worker spins up and detects the standard 80 COCO classes (the system specifically cares about `person`, `cell phone`, `laptop`, and `tv`).

Two decoders exist in the code — `decodeYOLO` (legacy/raw YOLO output format) and `decodeYOLO26` (the streamlined 6-value-per-box format YOLO26 actually emits: `x1, y1, x2, y2, confidence, class_id`). The pipeline currently runs on the YOLO26 decoder, which skips the older anchor-grid decoding entirely — YOLO26 outputs already-resolved boxes, so there's no need to manually compute scores across an 80-class softmax per anchor like older YOLO versions required.

### 3.2 Adaptive frame throttling — the system slows down when it's calm, and speeds up when it's worried

This is one of the more interesting design decisions in the whole project: **the object-detection frame rate is not fixed — it's driven by the current risk level.**

```js
const fpsMap = { low: 2, medium: 4, high: 6, critical: 8 }
```

When everything looks normal, the worker only runs inference **2 times per second** — plenty to catch a sustained absence or a phone left on the desk, while keeping CPU/GPU usage low for the entire duration of an exam. The moment the risk engine's score crosses into `medium`/`high`/`critical` territory, the main thread sends a `riskLevel` message back down to the worker, which **doubles, triples, or quadruples its own sampling rate** — going from checking twice a second to **8 times a second** as things get more suspicious.

This is a closed feedback loop: the risk engine watches the detector, and the detector listens back to the risk engine. The system effectively says: *"I don't need to work hard when nothing's wrong — but the instant something might be wrong, scrutinize the candidate far more closely."*

### 3.3 Frame management — never falling behind, never queueing stale frames

Because inference is the slowest step in the whole pipeline, the worker is built so it **only ever processes the most recent frame, never a backlog**:

```js
if (latestFrame) {
    overwrittenFrames++;
    latestFrame.close();   // drop the previous unprocessed frame
}
latestFrame = incomingFrame;
```

If a new frame arrives before the model has finished with the last one, the old frame is simply discarded (and properly `.close()`'d to free its underlying GPU/memory resource — `VideoFrame` objects are not garbage-collected automatically and must be explicitly released). This means the pipeline is always reacting to *now*, never processing a 2-second-old frame just because it happened to arrive first in a queue. Metrics are tracked for received vs. processed vs. overwritten frames, logged every 15 seconds, so performance can be observed live in the console.

### 3.4 Preprocessing — letterboxing to a square input

Webcam frames are rarely square, but YOLO26s expects a fixed 640×640 input. Rather than stretching the image (which would distort object proportions and hurt detection accuracy), the preprocessor **letterboxes**: it scales the frame down to fit inside 640×640 while preserving aspect ratio, then pads the remaining space with black, centering the real image:

```js
const scale = Math.min(MODEL_SIZE / iw, MODEL_SIZE / ih);
const dx = (MODEL_SIZE - nw) / 2;   // center horizontally
const dy = (MODEL_SIZE - nh) / 2;   // center vertically
```

The result is then converted from interleaved RGBA pixel data into the planar, normalized `[1, 3, 640, 640]` float tensor the model expects (each channel separated, values scaled to 0–1) — all done manually with a pre-allocated `Float32Array` buffer that's reused every frame rather than reallocated, avoiding repeated garbage-collection pressure on a hot path that runs multiple times per second.

### 3.5 Non-Maximum Suppression (NMS)

Object detectors typically propose multiple overlapping boxes for the same real-world object. The `nms()` function cleans this up using classic IoU-based suppression: sort all detections by confidence, greedily keep the highest-confidence box, then discard any remaining box that overlaps it (`IoU ≥ 0.45`) — repeating until no boxes remain. The `iou()` helper computing intersection-over-union is also reused throughout the codebase for the person-tracking logic described next.

---

## 4. Person Tracking & Identity Locking — Remembering *Who* the Candidate Is, Frame to Frame

This is the centerpiece of the detection pipeline, and it solves a problem that's easy to underestimate: **YOLO has no concept of identity.** Every single frame, it just returns a fresh batch of unlabeled boxes — "person here, person there." It has no idea if the "person" box in this frame is the *same* human as the "person" box from the frame before. Without something layered on top, the system can't reliably say "the candidate moved" versus "a different person walked into view," and every other system downstream — the gaze tracker, the motion detector's exclusion zone — would be working blind.

`getLockedPersonBox()` solves this with a lightweight, purpose-built tracker:

### 4.1 Locking onto a person

The first time a person is seen, the tracker simply locks onto whichever detected `person` box has the highest confidence score. From that point on, it isn't just looking for "a person" anymore — it's trying to keep following *that specific person* across every subsequent frame.

### 4.2 Re-identification by spatial overlap, not appearance

There's no facial recognition or re-ID embedding model here — re-identification is done purely through **geometry**. On every new frame, the tracker computes IoU (intersection-over-union) between the previously-locked box and every newly detected person box, and picks whichever one overlaps the most:

```js
for (const p of persons) {
  const overlap = iou(_lockedPersonBox, p.box)
  if (overlap > bestIou) { bestIou = overlap; bestBox = p.box }
}
```

The insight here is that a real person can't teleport between frames — at 2–8 frames per second, their bounding box in frame *N+1* will almost always still substantially overlap their box from frame *N*. So "whoever overlaps my last known position the most" is a cheap, fast, and surprisingly reliable proxy for "the same person I was already tracking" — without ever needing to run a second, heavier re-identification model.

A `MIN_BOX_AREA` filter (2% of frame area) also deliberately ignores tiny, distant, or partially-visible person detections when looking for a match — these are far more likely to be noise (a reflection, a partial limb at the frame edge) than a genuine candidate, and including them would make the lock unstable.

### 4.3 Confidence-weighted exponential smoothing — the "previous bbox" memory

This is the part worth calling out specifically, because it's a genuinely thoughtful piece of engineering: **the locked box doesn't just snap to the new detection — it blends the old position and the new one, and how much it blends is itself dynamic.**

```js
const a = Math.min(0.70, 0.15 + (bestIou - LOCK_IOU_MIN) * 1.1)
_lockedPersonBox = {
  x1: _lockedPersonBox.x1 * (1-a) + bestBox.x1 * a,
  y1: _lockedPersonBox.y1 * (1-a) + bestBox.y1 * a,
  x2: _lockedPersonBox.x2 * (1-a) + bestBox.x2 * a,
  y2: _lockedPersonBox.y2 * (1-a) + bestBox.y2 * a,
}
```

This is an exponential moving average (EMA) over the box coordinates — the new locked position is a weighted blend of *where the person was* and *where they appear to be now*. The blend factor `a` (alpha) isn't a fixed constant; it scales directly with how confident the match is:

| Match quality (IoU) | Smoothing weight (`a`) | Effective behavior |
|---|---|---|
| 0.25 (barely qualifies) | ~0.15 | Mostly trust the *old* position — a weak/ambiguous match shouldn't be allowed to yank the box around |
| 0.75 (strong overlap) | ~0.55 | Trust the new detection significantly — this is clearly the same person, in roughly the same place |
| 1.00 (perfect overlap) | ~0.70 | Follow quickly — no ambiguity, no need to lag behind real movement |

In other words: **the more certain the system is that it's still looking at the same person, the faster it's willing to update their tracked position. The less certain, the more it leans on history.** This is exactly the kind of self-correcting confidence-weighted filter used in real tracking systems (think Kalman-filter-adjacent logic, implemented here in a much lighter, purpose-built form) — it prevents the box from jittering wildly on noisy single-frame detections while still being responsive to genuine, sustained movement.

### 4.4 Tolerating temporary disappearance — "miss frames" instead of instant resets

People don't always face the camera perfectly, and detectors occasionally miss a frame even when the subject hasn't moved. If no detection overlaps the lock well enough, the tracker doesn't immediately give up — it increments a miss counter and **keeps returning the last known (stale) box** so downstream consumers (like the motion detector's exclusion zone, see §6) don't lose track of where the candidate roughly is. Only after `LOCK_MISS_MAX` (10) consecutive missed frames does the system conclude the person has genuinely left and clear the lock, ready to re-acquire fresh.

This "grace period" design is the difference between a system that's annoyingly trigger-happy (flagging "person missing" the instant someone leans down to tie their shoe) and one that correctly distinguishes a momentary occlusion from an actual departure.

### 4.5 Why this matters beyond just drawing a box

The locked person box isn't just a UI nicety — it's load-bearing infrastructure consumed by two other systems entirely:

- **The motion detector** (`motion-worker.js`) uses it to carve the candidate's own body out of the region it scans for background motion, so the candidate naturally moving in their chair never gets misread as "something moved in the background."
- **The gaze tracker** (`gaze-worker.js`) uses it to reject face detections that don't fall inside the locked person's region — preventing a second face (visible in a mirror, a photo, or a person walking past in the background) from being mistaken for the candidate's own gaze.

A single, well-engineered tracking primitive quietly underpins two completely different detection subsystems.

---

## 5. Phone & Screen-State Detection — Not Just "Is There a Phone," but "Is It Actually Being Used"

Detecting *that* a phone is present is the easy 80%. The genuinely hard, and far more useful, question is: **is the screen on?** A phone lying face-down and untouched on the desk is a completely different risk level from a phone being actively read. ProctorSense answers this without any extra model — purely through computer-vision analysis of pixel brightness within the detected phone's bounding box.

### 5.1 The aspect-ratio + confidence gate

Before even checking brightness, the system has to decide which `cell phone` detections are worth trusting. Raw YOLO confidence alone isn't always reliable for small, handheld objects, so `classifyPhone()` and the underlying `phoneSignal()` apply a secondary, geometry-based sanity filter:

```js
if (d.score >= CONFIG.HIGH_THRESHOLD) return true   // trust high-confidence hits outright
if (touchesFrameEdge(d.box)) return true             // partially-visible phones can't be ratio-checked fairly
const ratio = bw / bh
return (ratio > 0.20 && ratio < 0.75) ||   // portrait phone
       (ratio > 1.5  && ratio < 2.2)        // landscape phone
```

A genuine code comment in the source explains *why* this exists and why it isn't applied too strictly: an earlier, stricter version of this rule was **silently rejecting valid phones** held at an angle or partially covered by a hand — even when YOLO had correctly classified them. So the rule was loosened: any high-confidence detection bypasses the shape check entirely, and the aspect-ratio filter only kicks in to help validate borderline/lower-confidence detections. That's a deliberate, considered trade-off between false positives and false negatives, not an oversight.

### 5.2 Brightness-based screen-state classification — the clever part

Once a trustworthy phone box exists, `getScreenBrightnessState()` analyzes the **luminance pattern inside that box** to determine if the screen is actually lit:

```js
// Split box into center region (inner 50%) vs border region (outer ring)
const cx1 = x1 + bw * 0.25, cx2 = x1 + bw * 0.75   // ...etc
```

It computes three separate brightness averages every frame:

1. **Center brightness** — the luminance of the inner 50% of the phone's bounding box (where a screen would be, away from the bezel/case edges)
2. **Border brightness** — the luminance of the outer ring of the box (bezel, case, fingers, background bleed)
3. **Scene brightness** — the average luminance of the *entire 640×640 frame*, used as a baseline for "how bright is this room overall"

From these three numbers it derives two ratios that, together, distinguish a genuinely lit screen from a bright *reflection* of something else:

- **`sceneRatio`** = center brightness ÷ full-frame brightness → *is the phone notably brighter than the room around it?*
- **`uniformity`** = center brightness ÷ border brightness → *is the brightness evenly spread, or concentrated in one patchy spot?*

The key insight, stated directly in the code: **a live screen is uniformly bright across its whole surface (center ≈ border), while a reflection or glare is patchy — bright in one localized spot, dark elsewhere.** This lets the system tell apart "the phone's screen is genuinely on" from "the phone case caught a glint of overhead light":

```js
if (uniformity < 1.8 && (sceneRatio > 0.75 || centerBrightness > 80)) return 'on'
if (sceneRatio > 0.75 && uniformity >= 1.8) return 'unclear'   // bright but patchy = reflection
return 'off'                                                     // dim and patchy = off
```

This three-way outcome (`on` / `off` / `unclear`) feeds directly into incident severity: a phone with its screen **on** is classified `HIGH` severity (someone is actively reading/using it), screen **off** is `LOW` (present, but arguably idle), and `unclear` sits in between — all without ever needing a second machine-learning model. It's a pure signal-processing solution to what looks at first like a classification problem.

### 5.3 The same trick, reused for laptops and TVs

Exactly the same brightness-classification function powers `classifyDevice()` for laptops and TVs detected in frame — so a laptop with its lid open and screen lit gets flagged more seriously than one sitting closed on a shelf, using the identical center-vs-border-vs-scene brightness comparison.

### 5.4 Frame-edge tolerance

Both phone and device classification treat boxes touching the frame edge (`touchesFrameEdge()`) more leniently — a phone or laptop that's only partially visible at the very edge of the webcam's view is downgraded to `LOW` severity regardless of its detected confidence, since a partial, cut-off view is inherently less reliable evidence than a fully-framed detection.

---

## 6. Evidence-Based Condition Tracking — Treating Suspicion as Something That Accumulates and Decays

Rather than treating each frame's detections as an instant trigger ("phone detected → alarm now"), `worker.js` runs every signal through a generic `ConditionTracker` class that models suspicion the way a careful human observer would: **building confidence gradually as evidence accumulates, and losing it gradually once the evidence stops.**

### 6.1 Per-class confidence smoothing

Before a tracker even evaluates a condition, raw per-frame detection scores are smoothed using their own EMA:

```js
this.smoothedConf[det.class] =
    CONFIG.CONF_SMOOTH_ALPHA * det.score +
    (1 - CONFIG.CONF_SMOOTH_ALPHA) * prev;
```

And critically, classes that *stop* appearing don't just vanish from the smoothed-confidence map — they **decay** (`*= 0.6` per frame by default) rather than dropping to zero instantly. This means a phone that disappears for one single frame (occluded by a hand mid-movement, say) doesn't immediately reset the system's "belief" that a phone is present back to zero.

### 6.2 Evidence accumulation with frame-rate normalization

Each tracker maintains an `evidence` score (0–10) that rises when its signal function returns a positive value, and decays exponentially otherwise:

```js
this.evidence = Math.min(MAX_EVIDENCE, Math.max(0,
    this.evidence * Math.pow(DECAY_FACTOR, frameWeight) +
    signal * ACCUM_WEIGHT * frameWeight
));
```

The `frameWeight` term is a subtle but important detail: it normalizes for *actual elapsed time* between frames, not just frame count. Because the detection FPS is adaptive (§3.2) and can swing from 2 to 8 frames per second depending on risk level, evidence has to accumulate and decay at a **rate-independent** pace — otherwise running inference more often (when risk is already elevated) would itself cause evidence to build up faster purely as a side effect of sampling more, rather than because anything actually changed.

### 6.3 Hysteresis state machine — entering and exiting states at different thresholds

Each tracker sits in one of three states — `OK`, `YELLOW`, `RED` — and the thresholds for *entering* a worse state are deliberately higher than the thresholds for *exiting* it:

```js
THRESHOLDS: {
    YELLOW_ENTER: 4.0,  RED_ENTER: 7.5,
    RED_EXIT: 5.0,      YELLOW_EXIT: 2.0,
}
```

This is **hysteresis** — the same pattern used in thermostats — and it exists specifically to prevent flapping. Without it, evidence sitting right at a threshold boundary would cause the state to flicker rapidly between `OK` and `YELLOW` on every small fluctuation, generating a confusing stream of open/close incident noise. With separate enter/exit thresholds, a condition has to clear meaningfully past the boundary in either direction before the state actually changes, giving every state transition real, sustained backing.

### 6.4 Signal functions — purpose-built logic per condition

Four trackers run simultaneously, each with its own signal function tuned to its specific failure mode:

- **`PERSON_MISSING`** — returns full signal strength only when smoothed person-confidence drops below the low threshold; partial visibility (someone half-leaned out of frame) produces a partial, not binary, signal.
- **`EXTRA_PERSON`** — distinguishes a confidently-detected second person (signal `1.0`) from a low-confidence "partial person" that's "consistently there" but unclear (signal `0.3`) — e.g. someone visible at the very edge of frame.
- **`PHONE`** / **`DEVICE`** — feed off the aspect-ratio-gated detections described in §5, using smoothed confidence against the same high/low thresholds.

---

## 7. Gaze & Eye Tracking (`gaze-worker.js`) — Head Pose *and* Iris Position, Independently Calibrated

This is the most mathematically dense module in the project. It runs Google's **MediaPipe FaceLandmarker** (468-point face mesh + iris landmarks, GPU-accelerated, running entirely client-side from a locally-cached or CDN-fallback WASM bundle) and turns raw facial geometry into a real-time attentiveness signal.

### 7.1 Personal baseline calibration — because everyone's "neutral" face is different

The system doesn't compare head pose against some fixed universal "looking forward" angle — it spends the **first 30 frames** silently building a personal baseline for *this specific candidate's* natural resting head position:

```js
baseline = {
  pitch: median(calibFrames.map(f => f.pitch)),
  yaw:   median(calibFrames.map(f => f.yaw)),
  roll:  median(calibFrames.map(f => f.roll)),
}
baselineMAD = { /* median absolute deviation per axis */ }
```

Using the **median** rather than the mean is a deliberate robustness choice — a single frame where the candidate glances away during calibration won't drag a mean baseline off-center the way it could skew an average, since the median is far more resistant to outliers.

### 7.2 Two-stage calibration — head pose, then a silent second pass for eye position

Once head-pose calibration locks in, the system automatically starts a **second, silent calibration phase** purely for natural eye/iris resting position — but only accepts frames for this second pass where the head is already confirmed stable (`headDeviation < SOFT_THRESHOLD`). This stops a moving head from corrupting the eye baseline, since iris position relative to the eye socket only means something consistent when the head itself isn't also moving.

### 7.3 Iris tracking — computing gaze direction from raw landmarks

`computeEyeGaze()` derives horizontal/vertical gaze direction purely from geometric ratios between the iris center landmark and the four corners of the eye socket — no separate gaze-estimation model required:

```js
const leftGazeX = (leftIris.x - leftOuter.x) / leftSpanX  // 0 = outer corner, 1 = inner corner
const gazeX = ((leftGazeX + rightGazeX) / 2) * 2 - 1       // normalize to -1..1, 0 = centered
```

Both eyes are averaged together, then the result is run through a **6-frame rolling-average smoothing buffer** (`IRIS_SMOOTH_FRAMES`) to suppress the natural jitter of iris-position estimation frame to frame. There's also a built-in sanity check: if eye-openness (the ratio of vertical eye-socket span) drops below 0.15, the function assumes the eyes are closed or the estimate is unreliable, and returns `null` rather than feeding noisy data downstream — blinking shouldn't register as "looking somewhere."

### 7.4 Corroboration logic — combining head pose and eye direction intelligently

This is the most elegant piece of logic in the gaze module. `computeDeviation()` doesn't just add head-deviation and eye-deviation together — it checks whether they **agree**:

```js
const corroborated =
  (eyeRight && headRight) || (eyeLeft && headLeft) ||
  (eyeDown && headDown)   || (eyeUp && headUp)

if (corroborated) return base * 1.8   // head AND eyes agree → strong signal, amplify
// Eyes alone — meaningful independent contribution (50% weight)
const eyeContribution = eyeMagnitude * 15 * 0.5
return base + eyeContribution
```

If the head turns right *and* the eyes also look right, that's strong, corroborated evidence of genuinely looking away — so the deviation score gets amplified by 1.8×. But if only the eyes move (head stays still, eyes dart to the side — exactly the behavior someone glancing at a hidden note or second screen *without* turning their head would produce), that's treated as meaningful evidence on its own, contributing at half-weight rather than being ignored just because the head didn't move. This means the system is specifically designed to catch the subtle "eyes-only" cheating pattern that head-pose-only proctoring tools would completely miss.

### 7.5 State machine with separate grace periods per state

Gaze state (`focused → glancing → away`, or `→ absent` if no face at all) uses time-based grace periods rather than instant transitions:

- A deviation has to stay elevated for `GLANCE_GRACE_MS` (2000ms) before counting as a full "away" event rather than a brief glance.
- A missing face has to stay missing for `ABSENT_GRACE_MS` (1000ms) before being called "absent" — a single dropped frame from the face detector doesn't instantly read as the candidate having left.

### 7.6 Sustained-gaze "eye alert" detection with episode-based re-arming

Beyond the basic away/focused state, a second, independent layer (`checkEyeAlerts`) specifically watches for **prolonged eye movement while the head appears to stay "focused"** — catching scanning/reading behavior that the coarser state machine wouldn't flag on its own. It tracks a continuous "episode" of elevated eye magnitude and re-arms its counter every `EYE_SUSTAIN_MS` (1500ms) *within* that same episode, so a long, sustained stare keeps accumulating evidence over time rather than only counting once at the start:

```js
if (now - lastEyeGazeCountAt >= EYE_SUSTAIN_MS) {
  eyeAlertEvents.push(now)
  lastEyeGazeCountAt = now   // re-arm within the same episode
}
```

Once enough of these events accumulate within a rolling 60-second window, a graded alert fires (`low` / `medium` / `high` severity based on event count), with its own 30-second cooldown so it doesn't spam re-fire every frame while still above threshold.

### 7.7 Identity guard — rejecting faces that aren't the locked candidate

As mentioned in §4.5, the gaze worker receives the locked person bounding box from the object detector and uses it as a sanity filter: if a detected face's centroid falls clearly outside that box (with a 20% padding allowance for detection jitter), the frame is **ignored entirely** for gaze purposes. This stops a second face — visible in a photo, a mirror, or someone passing by — from ever being mistaken for the candidate's own gaze direction.

---

## 8. Background Motion Detection (`motion-worker.js`) — A Hand-Built Computer Vision Pipeline

This worker doesn't use a neural network at all — it's a classic, hand-rolled computer-vision pipeline (the kind you'd traditionally reach for OpenCV to build), reimplemented from scratch in plain JavaScript so it can run entirely in a Web Worker with zero native dependencies. It's explicitly noted in the comments as a port of an existing Python/OpenCV reference implementation (`app.py`), down to matching the exact grayscale conversion weights OpenCV uses, so behavior stays consistent between the two.

### 8.1 The pipeline, phase by phase

1. **Decode** the `VideoFrame` to raw RGBA pixels via a reused `OffscreenCanvas`.
2. **Build a candidate mask** — the region of the frame allowed to register motion (see §8.2 below).
3. **Convert to grayscale** using OpenCV-matching luminance weights (`0.299·R + 0.587·G + 0.114·B`).
4. **Frame-difference** the current grayscale frame against the previous one, thresholded at a brightness delta of 30, masked to the candidate region.
5. **Morphological filtering** — erode-then-dilate ("open") to remove tiny noise specks, followed by dilate-then-erode ("close") to fill small holes in larger real motion blobs. Both are implemented as hand-written 5×5 structuring-element kernels.
6. **Connected-component labeling** via 8-connected BFS flood fill, keeping only blobs whose pixel area exceeds `MIN_BLOB_AREA` (1500px) — discarding sub-threshold noise entirely.
7. **Zone mapping & alert suppression** (§8.3).
8. **Diff visualization** — a red-tinted overlay image highlighting exactly which pixels changed, for the reviewer-facing comparison modal.

### 8.2 Exclusion masking — reusing the locked person box to avoid self-triggering

This is where the person-tracking system from §4 pays off again. `buildCandidateMask()` takes the **locked person bounding box** passed in from the main thread and explicitly **zeroes out** that region from the motion-detection mask — the system is told *"do not look for motion here, this area is the candidate's own body."*

```js
const bboxWidthRatio = (right - left) / width
const padFactor = bboxWidthRatio > 0.6 ? 0 : bboxWidthRatio > 0.35 ? 0.10 : 0.15
const padX = Math.floor((right - left) * padFactor)
```

The padding around the excluded box is itself adaptive — a candidate who fills most of the frame (close to the camera) gets little to no extra padding, while a candidate who's smaller in frame gets proportionally more padding around their box. This accounts for the fact that a smaller, more distant person has comparatively more "fringe" movement around their silhouette (loose clothing, hair, arm gestures extending slightly past the box) that shouldn't count as suspicious background motion.

Without this exclusion zone, every normal head turn or hand gesture by the candidate themselves would register as "background motion" — this single mechanism is what makes the whole motion detector usable at all in practice rather than firing constantly.

### 8.3 Zone-based alert suppression — cooldown per region, not globally

Rather than a single global "motion happened, alert!" cooldown, `shouldAlertForZones()` tracks the **last alert time per zone** independently (the frame is divided into a 2×3 grid: top/bottom × left/center/right). Each zone gets its own 30-second suppression window:

```js
for (const zone of activeZones) {
  if (!recentZoneAlerts.has(zone)) {
    recentZoneAlerts.set(zone, now)
    return true   // first alert for THIS zone, fire it
  }
}
return false   // every active zone already alerted recently
```

This means continuous motion in one part of the frame (a fan, a flickering light in the top-right) won't spam repeated alerts, while genuinely *new* motion appearing in a *different* zone — even moments later — still gets through immediately, since that zone hasn't been "used up" by its own cooldown yet.

### 8.4 Deliberately slow sampling, by design

Unlike the object detector, this worker is fed frames on a fixed 5-second interval rather than continuously (`MOTION_SAMPLE_INTERVAL` in the main script). Background motion worth flagging — someone walking past, a second screen flickering on — is, by definition, the kind of event that's still detectable a few seconds later. Sampling this infrequently keeps the heavier per-pixel morphology/connected-components work (which runs over a full-resolution frame, not a downscaled 640×640 like the object detector) cheap over the life of a long exam session.

### 8.5 Efficient memory reuse

Every intermediate buffer (`grayPrev`, `grayCurr`, `diffMask`, `candidateMask`, `filteredMask`, `validMask`, `morphScratch`) is allocated exactly once per resolution and reused on every subsequent frame via `ensureBuffers()` rather than reallocated — a deliberate performance choice for a long-running worker that needs to avoid GC churn over a session that could run for hours.

---

## 9. The Risk Engine (`risk-engine.js`) — Turning Five Independent Signals Into One Trustworthy Score

Every detection system above produces its own raw signal. The risk engine's job is to combine all of them into a single, slow-moving, hard-to-game risk score (0–100), decide when that score justifies raising an actual incident, and manage the entire lifecycle of that incident from creation to resolution.

### 9.1 Windowed scoring — ratios, not single frames

Every 5 seconds (`SCORE_INTERVAL_MS`), the engine looks back over the *ratio* of frames in that window where each condition was true — not single-frame snapshots:

```js
const absentRatio = signals.faceAbsentCount / total
if (absentRatio > 0.5) score += 40
else if (absentRatio > 0.2) score += 20
```

A candidate who briefly leaves frame for one frame out of a hundred barely moves the score; a candidate absent for half the window contributes meaningfully. Five separate ratios (absence, multiple-faces, phone, device, gaze-away) each contribute points on their own threshold curve, summed and capped at 100. The **gaze** signal additionally blends in a continuous component — not just binary away/not-away — by averaging the raw 0–100 gaze score across the window and folding in up to 30 extra points, capturing nuance (sustained partial deviation, oscillation patterns) that a simple ratio would flatten away.

### 9.2 Risk accumulation and decay — asymmetric by design

The headline risk number doesn't track the windowed score directly — it **rises quickly toward bad news and decays slowly back down**, using two completely different mathematical models for the two directions:

```js
function accumulateRisk(windowScore, dt) {
  const pull = RISK_ACCUMULATE_RATE * (dt / SCORE_INTERVAL_MS)
  currentRisk = Math.min(100, Math.max(windowScore, currentRisk + windowScore * pull))
}
function decayRisk(dt) {
  const decayPerMs = Math.pow(0.5, 1 / RISK_DECAY_HALF_LIFE_MS)
  currentRisk = currentRisk * Math.pow(decayPerMs, dt)
}
```

Accumulation pulls the risk score toward the current window's score at a controlled rate (35% per interval) — fast enough to react, but not so fast that one noisy window spikes the number instantly. Decay, by contrast, is a true **exponential half-life** (3 seconds) — meaning risk doesn't just fade, it fades at a *mathematically consistent, predictable rate* regardless of how high it climbed, the same model used to describe radioactive decay or drug elimination in pharmacology. The practical effect: a single moment of high suspicion sends the number up immediately, but it takes a real, sustained period of "all clear" for the number to fully settle back down — exactly the asymmetry you'd want from a system meant to be cautious rather than dismissive.

### 9.3 Adaptive sampling rate, downstream of risk

Just like the object detector's FPS (§3.2), the risk engine's own UI-facing sampling interval also tightens as risk rises — from checking every 2 seconds when calm down to every 200ms once risk exceeds 90. This keeps the dashboard feeling instantly responsive exactly when responsiveness matters most, without burning cycles refreshing a calm, unchanging UI 10× more often than necessary.

### 9.4 Per-category incident confirmation — requiring sustained evidence before flagging anything

`evaluateIncident()` runs an independent confirmation counter for each incident category (PHONE, DEVICE, GAZE, MULTI, PRESENCE). A category only actually triggers a visible incident after **8 consecutive confirming ticks** (`CATEGORY_CONFIRM_TICKS`) — and once resolved, it can't re-trigger again for a full 60-second cooldown (`CATEGORY_COOLDOWN_MS`). This is the same anti-flapping philosophy as the hysteresis state machine in §6.3, applied at the incident-lifecycle level instead of the raw-signal level — there are effectively two independent layers of "are you sure?" between a raw detection and a visible flag on the dashboard.

### 9.5 Correlation — escalating severity when multiple bad signals overlap

`triggerIncident()` checks whether *other* incident categories are already active at the moment a new one fires. If so, the new incident's severity is automatically escalated to `HIGH` and its description is annotated to note the overlap:

```js
const otherActive = Object.keys(activeIncidents).filter(c => c !== incident.category)
const correlated = otherActive.length > 0
const severity = correlated ? 'HIGH' : incident.severity
```

A phone appearing on its own might be `MEDIUM`. A phone appearing *at the same time as* the candidate going briefly off-camera is treated as meaningfully more suspicious than either signal alone — `generateFlagContext()` has explicit, hand-written rules for several of these correlated patterns (phone + absence, multiple faces + phone, multiple faces + device), each described in its own tailored, human-readable explanation rather than a generic "multiple flags" message.

### 9.6 Face-position drift detection — a second, independent behavioral signal

Separately from gaze, the risk engine tracks the **raw screen-space position of the candidate's face centroid** over time. After a 30-frame calibration period establishes a "home" position (again using the median, for the same outlier-resistance reason as §7.1), sustained drift away from that home position — held for at least 8 seconds (`DRIFT_SUSTAIN_MS`) — triggers a low-severity behavioral flag describing which direction the candidate shifted and by how much. This catches a different failure mode than gaze tracking: a candidate who has physically *moved* (leaned to the side, shifted their chair) rather than one who's merely looking elsewhere while staying in the same seated position.

### 9.7 Proof capture and the incident lifecycle

Every triggered incident captures a snapshot of the live video at the moment of triggering (`window.captureProof()`), stored as a transferable `ImageBitmap`. Incidents persist through three states — opening (with an "onset" snapshot), staying active, and resolving (re-snapshotted as the "best frame," sometimes a clearer shot than the triggering moment) — each transition pushed into both the live "Active Flags" panel and a permanent "Incident History" sidebar that survives even after the active flag clears, so a full record of the session remains reviewable afterward.

### 9.8 Proctor controls — escalation, dismissal, and JSON export

Flags surfaced to the live panel carry **Escalate** and **Dismiss** actions, letting a human reviewer make the final call on any automated flag rather than treating the system's output as unappealable — escalating promotes a flag into the permanent incident log with a "MANUAL" badge, while dismissing visually marks it resolved without escalation. At the end of a session, `exportIncidentReport()` serializes every recorded flag — category, severity, confidence, explanation, timestamp, and a compressed JPEG snapshot — into a downloadable JSON file, giving the proctor a complete, portable audit trail of the entire session.

---

## 10. Capture Pipeline — Getting Frames From the Webcam to Three Workers Without Copying

The main thread's job is mostly plumbing, but it's carefully-built plumbing:

### 10.1 `MediaStreamTrackProcessor` — bypassing the canvas entirely

Rather than the traditional approach of drawing video frames to a `<canvas>` and reading pixels back out (which incurs a decode → GPU upload → CPU readback round trip every frame), the capture loop uses the modern **`MediaStreamTrackProcessor`** API to pull raw `VideoFrame` objects directly off the camera's media track:

```js
const processor = new MediaStreamTrackProcessor({ track });
const reader = processor.readable.getReader();
```

These frames are then handed to workers using **Transferable objects** (`[frame]` as the second argument to `postMessage`) rather than copied — ownership of the underlying frame buffer moves to the worker with zero memory-copy cost, which matters enormously when you're moving video frames around multiple times per second.

### 10.2 Capture-side throttling, independent of worker-side throttling

The main capture loop has its own throttle — a flat 8 FPS target (`TARGET_CAPTURE_FPS`) — separate from the adaptive 2–8 FPS the object-detection worker applies on its own side (§3.2). Frames arriving faster than the capture interval allows are dropped (and explicitly `.close()`'d) before they're ever sent anywhere, which keeps the *sending* side cheap regardless of the receiving worker's own internal pacing.

### 10.3 Fan-out to three destinations from one read loop

Every frame that survives the capture throttle is potentially cloned (via `new VideoFrame(frame)`, which creates an independent reference to the same underlying frame data rather than a full pixel copy) and routed to up to three places: the object-detection worker (every surviving frame), the motion worker (only once every 5 seconds), and — separately, downstream of the object detector's own response — the gaze worker (throttled to roughly every 100ms). Each destination gets exactly the cadence it needs, from a single shared capture loop.

---

## 11. The Dashboard — Making All of This Legible to a Human

A sophisticated backend is only as useful as the interface built on top of it, and `index.html` invests heavily here.

### 11.1 Two audiences, two views

The dashboard deliberately separates **what a proctor watching live needs** from **what an engineer auditing the system needs**:

- The **main view** is intentionally minimal: a risk ring, a belief-bar panel for each signal, a live timeline chart, an active-flags panel, and an incident history sidebar — everything a non-technical proctor needs to glance at and understand instantly.
- The **Advanced Analytics modal** is a completely separate, far denser view: per-signal rolling history graphs with threshold lines drawn directly on them, a live "Explain Score" breakdown showing exactly how many frames contributed how many points to the current risk number, per-signal incident counts and timestamps, and hoverable snapshot thumbnails for every recorded event.

### 11.2 The "Explain Score" panel — radical transparency into the scoring math

Rather than asking the proctor to trust a single opaque number, the score explainer renders the *actual arithmetic* behind the current risk score in real time — frame counts, percentages, and exactly how many points each signal is contributing this window, pulled live from the risk engine's own internal counters:

```
Person Absent     12/40 frames (5s) · 30% → +20 pts
Phone Visible      3/40 frames (5s) ·  7% → no contribution
```

This turns the risk score from a black box into something a human reviewer can audit and sanity-check at a glance.

### 11.3 Animated, perceptually-tuned risk display

The headline risk number doesn't just snap to its new value — it eases toward it using different animation curves depending on direction: a fast cubic ease-out when rising (140ms — risk increases should *feel* immediate) and a slower, gentler ease when falling (650ms — a sense of "settling down" rather than an abrupt drop), reinforcing the same "quick to worry, slow to relax" philosophy baked into the underlying math in §9.2.

### 11.4 Risk cycle counter — counting meaningful escalation events, not noise

A "risk cycle" increments specifically when the score rises by more than 25 points from its post-reset baseline and then falls back — a deliberately coarse-grained way of counting "how many times did something genuinely concerning happen and then resolve" over a session, distinct from the constant minor wobble of the raw score.

### 11.5 Self-contained, additive architecture

Two of the dashboard's script blocks are explicitly written as **non-invasive overlays** on top of the original inline scripts — they read already-public state (`window._getRisk()`, belief-bar text content, DOM mutations on existing elements) rather than modifying the original logic directly, and in a couple of places **monkey-patch** existing `window`-exposed functions (`setBeliefBar`, `addFlag`, `updateRiskTimeline`) by wrapping the original implementation and calling through to it, then layering extra behavior (like keeping the score-explainer panel in sync) on top — a pattern that let the UI grow substantially in complexity over time without needing to touch or risk breaking the original, already-working core logic.

### 11.6 Live diagnostic surfaces for engineers

Beyond the analytics modal, several smaller diagnostic tools are wired directly into the camera card itself: a live eye-tracking debug panel (raw yaw/pitch/iris bars), a one-click gaze recalibration button, and a persistent phone-screen-state indicator (`ON`/`OFF`/`–`) sourced directly from the brightness classifier in §5.2 — letting anyone watching the session verify the underlying detectors are behaving correctly, not just trust the final risk number.

---

## 12. Summary — Everything at a Glance

| Module | Core technology | Standout technique |
|---|---|---|
| **worker.js** | YOLO26s via ONNX Runtime Web (WebGPU) | Risk-adaptive frame rate; confidence-weighted box smoothing for person locking; brightness-ratio screen-state classification |
| **gaze-worker.js** | MediaPipe FaceLandmarker (468-pt mesh + iris) | Personal median-baseline calibration; head/eye corroboration logic; episode-based sustained-stare detection |
| **motion-worker.js** | Hand-rolled frame-differencing CV pipeline | Person-box exclusion masking with adaptive padding; per-zone alert cooldowns; connected-component blob filtering |
| **risk-engine.js** | Evidence accumulation + hysteresis state machines | Asymmetric accumulate-fast/decay-slow risk curve (half-life decay); cross-signal correlation escalation; face-drift tracking |
| **index.html** | Vanilla JS, Web Workers, Canvas 2D, transferable objects | Zero-copy frame fan-out via `MediaStreamTrackProcessor`; live "Explain Score" transparency panel; dual proctor/engineer views |

**The thread that ties all five modules together**: almost nothing in this system reacts to a single frame in isolation. Every signal — person presence, phone state, gaze, motion, the aggregate risk score itself — is smoothed, confirmed over a window, given hysteresis between its on/off thresholds, and allowed to decay gracefully rather than reset instantly. That consistent design philosophy, applied independently at five different layers of the stack, is what separates this from a simple "run a model, show the output" demo and makes it behave like a system actually built to be trusted over a real, multi-hour exam session.