importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.min.js");

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

let session = null;
let PROCESS_FPS = 2;
let minIntervalUs = 1_000_000 / PROCESS_FPS;

function updateProcessFPS(level) {
    const fpsMap = {
        low: 2,
        medium: 4,
        high: 6,
        critical: 8
    }
    PROCESS_FPS = fpsMap[level] || 2
    minIntervalUs = 1_000_000 / PROCESS_FPS
    //console.log("FPS updated to:", PROCESS_FPS)
}
let lastProcessedTs = 0;
let latestFrame = null;
let latestTimestamp = 0;
let processing = false;
const metrics = {
    received: 0,
    processed: 0,
    overwritten: 0,

    preprocessMs: [],
    inferenceMs: [],
    decodeMs: [],
    totalMs: [],
    frameAgeMs: []
};
let receivedFrames = 0;
let overwrittenFrames = 0;
let processedFrames = 0;
const FACE_VIS_WARMUP_FRAMES = 20;
let faceVisProcessedCount = 0;

async function loadModel() {
    session = await ort.InferenceSession.create(
    "./static-files/yolo26s.onnx",
    { executionProviders: ["webgpu"] }
    );

    postMessage("YOLOv26 model loaded");
}

loadModel().catch((err) => console.error(err));;

const labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];


const MODEL_SIZE = 640;

const canvas = new OffscreenCanvas(
    MODEL_SIZE,
    MODEL_SIZE
);

const ctx = canvas.getContext("2d", {
    willReadFrequently: true
});

const inputBuffer = new Float32Array(
    3 * MODEL_SIZE * MODEL_SIZE
);


self.onmessage = (e) => {

    const data = e.data;

    if (data.type === 'riskLevel') {
        updateProcessFPS(data.level)
        return
    }

    if (data.type !== "frame") return;

    receivedFrames++;
    metrics.received++;

    const incomingFrame = data.frame;
    const ts = incomingFrame.timestamp;

    if (ts - lastProcessedTs < minIntervalUs) {
        metrics.overwritten++;
        overwrittenFrames++;
        incomingFrame.close();
        return;
    }

    if (latestFrame) {
        overwrittenFrames++;
        metrics.overwritten++;
        latestFrame.close();
    }

    latestFrame = incomingFrame;
    latestTimestamp = ts;
    lastProcessedTs = ts;
};


function decodeYOLO(output, labels, confThreshold = 0.25) {

    const data = output.data;
    const [_, channels, numBoxes] = output.dims;

    const results = [];

    for (let i = 0; i < numBoxes; i++) {

        const x = data[0 * numBoxes + i];
        const y = data[1 * numBoxes + i];
        const w = data[2 * numBoxes + i];
        const h = data[3 * numBoxes + i];

        let bestClass = -1;
        let bestScore = 0;

        for (let c = 4; c < channels; c++) {

            const score = data[c * numBoxes + i];

            if (score > bestScore) {
                bestScore = score;
                bestClass = c - 4;
            }
        }

        if (bestScore >= confThreshold) {

            results.push({
                class: labels[bestClass],
                score: bestScore,

                box: {
                    x1: x - w / 2,
                    y1: y - h / 2,
                    x2: x + w / 2,
                    y2: y + h / 2
                }
            });
        }
    }

    return results;
}

 function decodeYOLO26(output, labels, confThreshold = 0.25) {
    const data = output.data
    const numBoxes = output.dims[1]
    const results = []

    for (let i = 0; i < numBoxes; i++) {
        const x1 = data[i * 6 + 0]
        const y1 = data[i * 6 + 1]
        const x2 = data[i * 6 + 2]
        const y2 = data[i * 6 + 3]
        const conf = data[i * 6 + 4]
        const classId = Math.round(data[i * 6 + 5])

        if (conf >= confThreshold) {
            results.push({
                class: labels[classId] || "unknown",
                score: conf,
                box: { x1, y1, x2, y2 }
            })
        }
    }

    return results
}

function iou(a, b) {

    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const inter =
        Math.max(0, x2 - x1) *
        Math.max(0, y2 - y1);

    const areaA =
        (a.x2 - a.x1) *
        (a.y2 - a.y1);

    const areaB =
        (b.x2 - b.x1) *
        (b.y2 - b.y1);

    return inter / (areaA + areaB - inter);
}

function nms(detections, iouThreshold = 0.45) {

    detections.sort((a, b) => b.score - a.score);

    const kept = [];

    while (detections.length) {

        const current = detections.shift();

        kept.push(current);

        detections = detections.filter(d =>
            iou(current.box, d.box) < iouThreshold
        );
    }
    return kept;
}



function safeClose(frame) {
    try {
        frame?.close();
    } catch (e) {
    }
}

function logStats() {

    postMessage({
        type: "stats",
        receivedFrames,
        overwrittenFrames,
        processedFrames
    });
}

setInterval(logStats, 15000);



async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function preprocess(frame) {
    const iw = frame.displayWidth;
    const ih = frame.displayHeight;

    const snapshotBitmap = await createImageBitmap(frame);

    const scale = Math.min(
        MODEL_SIZE / iw,
        MODEL_SIZE / ih
    );

    const nw = Math.round(iw * scale);
    const nh = Math.round(ih * scale);
    const dx = (MODEL_SIZE - nw) / 2;
    const dy = (MODEL_SIZE - nh) / 2;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);

    ctx.drawImage(
        frame,
        dx,
        dy,
        nw,
        nh
    );

    const imageData = ctx.getImageData(
        0,
        0,
        MODEL_SIZE,
        MODEL_SIZE
    );

    const data = imageData.data;

    let p = 0;

    for (let y = 0; y < MODEL_SIZE; y++) {
        for (let x = 0; x < MODEL_SIZE; x++) {
            const idx = (y * MODEL_SIZE + x) * 4;

            inputBuffer[p] =
                data[idx] / 255;

            inputBuffer[p + MODEL_SIZE * MODEL_SIZE] =
                data[idx + 1] / 255;

            inputBuffer[p + 2 * MODEL_SIZE * MODEL_SIZE] =
                data[idx + 2] / 255;

            p++;
        }
    }

    return {
        tensor: new ort.Tensor(
            "float32",
            inputBuffer,
            [1, 3, MODEL_SIZE, MODEL_SIZE]
        ),
        meta: { dx, dy, scale, timestamp: frame.timestamp },
        snapshotBitmap,
        imageData   // ← add this
    };
}


function avg(arr) {
    if (!arr.length) return 0;

    return (
        arr.reduce((a, b) => a + b, 0)
        / arr.length
    );
}

function resetMetrics() {

    metrics.received = 0;
    metrics.processed = 0;
    metrics.overwritten = 0;

    metrics.preprocessMs.length = 0;
    metrics.inferenceMs.length = 0;
    metrics.decodeMs.length = 0;
    metrics.totalMs.length = 0;
    metrics.frameAgeMs.length = 0;
}

const CONFIG = {
    HIGH_THRESHOLD: 0.70,
    LOW_THRESHOLD: 0.45,
    MAX_EVIDENCE: 10.0,
    DECAY_FACTOR: 0.80,
    ACCUM_WEIGHT: 2.5,
    CONF_SMOOTH_ALPHA: 0.4,
    THRESHOLDS: {
        YELLOW_ENTER: 4.0,
        RED_ENTER: 7.5,
        RED_EXIT: 5.0,
        YELLOW_EXIT: 2.0,
    }
};

class ConditionTracker {
    constructor(name, signalFn) {
        this.name = name;
        this.signalFn = signalFn;
        this.evidence = 0;
        this.state = 'OK';
        this.smoothedConf = {};
        this.lastFrameTime = null;
    }

    update(detections, now) {
        const dt = this.lastFrameTime ? (now - this.lastFrameTime) / 1000 : 1 / 3;
        const frameWeight = (1 / 3) / dt; // normalized to 3 FPS baseline
        this.lastFrameTime = now;

        // Update smoothed confidences — use the MAX score per class in this
        // frame, not whichever detection happens to be last in the array.
        // Otherwise a second low-confidence detection of the same class
        // (extra person, misclassified object, etc.) overwrites and dilutes
        // your own high-confidence detection in the same frame.
        const seen = new Set();
        const maxScoreByClass = {};
        for (const det of detections) {
            maxScoreByClass[det.class] = Math.max(maxScoreByClass[det.class] ?? 0, det.score);
        }
        for (const cls of Object.keys(maxScoreByClass)) {
            seen.add(cls);
            const prev = this.smoothedConf[cls] ?? maxScoreByClass[cls];
            this.smoothedConf[cls] =
                CONFIG.CONF_SMOOTH_ALPHA * maxScoreByClass[cls] +
                (1 - CONFIG.CONF_SMOOTH_ALPHA) * prev;
        }
        for (const cls of Object.keys(this.smoothedConf)) {
            if (!seen.has(cls)) this.smoothedConf[cls] *= this.confDecay || 0.6;
        }

        const signal = this.signalFn(detections, this.smoothedConf);

        this.evidence = Math.min(CONFIG.MAX_EVIDENCE, Math.max(0,
            this.evidence * Math.pow(CONFIG.DECAY_FACTOR, frameWeight) +
            signal * CONFIG.ACCUM_WEIGHT * frameWeight
        ));

        this.state = this._nextState(this.evidence, this.state);
        return this.state;
    }

    _nextState(ev, cur) {
        const T = CONFIG.THRESHOLDS;
        if (cur === 'OK') {
            if (ev >= T.RED_ENTER) return 'RED';
            if (ev >= T.YELLOW_ENTER) return 'YELLOW';
            return 'OK';
        }
        if (cur === 'YELLOW') {
            if (ev >= T.RED_ENTER) return 'RED';
            if (ev >= T.YELLOW_EXIT) return 'YELLOW';
            return 'OK';
        }
        // RED
        if (ev >= T.RED_EXIT) return 'RED';
        if (ev >= T.YELLOW_EXIT) return 'YELLOW';
        return 'OK';
    }
}

// Signal functions
const personMissingSignal = (dets, smoothed) => {
    const conf = smoothed['person'] ?? 0;
    if (conf >= CONFIG.HIGH_THRESHOLD) return 0.0;
    if (conf >= CONFIG.LOW_THRESHOLD) return 0.5;

    // Fallback: if the person lock broke (e.g. small detection box) but raw
    // detections still show a person, do NOT report "nobody visible".
    const highConfPersons = dets.filter(d => d.class === 'person' && d.score > 0.4);
    if (highConfPersons.length > 0) return 0.0;

    return 1.0; // nobody visible
};

const extraPersonSignal = (dets, smoothed) => {
  const persons = dets.filter(d => d.class === 'person');
  const highConf = persons.filter(d => d.score >= CONFIG.HIGH_THRESHOLD);
  if (highConf.length >= 2) return 1.0;
  
  // partial person — low confidence but consistently there
  const partialConf = persons.filter(d => d.score >= 0.25 && d.score < CONFIG.HIGH_THRESHOLD);
  if (partialConf.length >= 1) return 0.3;
  
  return 0.0;
};

function touchesFrameEdge(box, margin = 10) {
  return box.x1 < margin || box.y1 < margin ||
         box.x2 > MODEL_SIZE - margin || box.y2 > MODEL_SIZE - margin
}

const classifyPhone = (dets, imageData) => {
  const phoneDets = dets.filter(d => {
    if (d.class !== 'cell phone') return false
    if (d.score >= CONFIG.HIGH_THRESHOLD) return true
    const bw = d.box.x2 - d.box.x1
    const bh = d.box.y2 - d.box.y1
    if (touchesFrameEdge(d.box)) return true
    const ratio = bw / bh
    return (ratio > 0.20 && ratio < 0.75) || (ratio > 1.5 && ratio < 2.2)
  })
  if (phoneDets.length === 0) return null

  const best = phoneDets.reduce((a, b) => (b.score > a.score ? b : a))

  if (touchesFrameEdge(best.box) && best.score < CONFIG.HIGH_THRESHOLD) {
    return { category: 'PHONE', severity: 'LOW', confidence: best.score,
      explanation: 'Phone partially visible at frame edge' }
  }

const screen = getScreenBrightnessState(imageData, best.box)
  if (screen === 'on') {
    return { category: 'PHONE', severity: 'HIGH', confidence: best.score,
      explanation: 'Phone visible with active screen' }
  }
  if (screen === 'unclear') {
    return { category: 'PHONE', severity: 'MEDIUM', confidence: best.score,
      explanation: 'Phone visible — screen state unclear' }
  }
  return { category: 'PHONE', severity: 'LOW', confidence: best.score,
    explanation: 'Phone visible with screen off' }
}

const classifyDevice = (dets, imageData) => {
  const deviceDets = dets.filter(d => d.class === 'laptop' || d.class === 'tv')
  if (deviceDets.length === 0) return null

  const best = deviceDets.reduce((a, b) => (b.score > a.score ? b : a))

  if (touchesFrameEdge(best.box)) {
    return { category: 'DEVICE', severity: 'LOW', confidence: best.score,
      explanation: `${best.class} visible only at frame edge` }
  }

const screen = getScreenBrightnessState(imageData, best.box)
  if (screen === 'on') {
    return { category: 'DEVICE', severity: 'HIGH', confidence: best.score,
      explanation: `${best.class} visible with active screen` }
  }
  if (screen === 'unclear') {
    return { category: 'DEVICE', severity: 'MEDIUM', confidence: best.score,
      explanation: `${best.class} visible — screen state unclear` }
  }
  return { category: 'DEVICE', severity: 'LOW', confidence: best.score,
    explanation: `${best.class} visible with screen off` }
}

function classifyFaceVisibility(imageData, personBox) {
  if (!personBox) return null

  // Derive face crop: top 30% of the person box (head region),
  // inset 18% on the left/right edges to exclude rim-light/halo bleed
  // that hugs the boundary between head and background.
  const bw = personBox.x2 - personBox.x1
  const bh = personBox.y2 - personBox.y1
  const INSET = 0.30
  const fx1 = Math.max(0, Math.floor(personBox.x1 + bw * INSET))
  const fy1 = Math.max(0, Math.floor(personBox.y1))
  const fx2 = Math.min(MODEL_SIZE - 1, Math.ceil(personBox.x1 + bw * (1 - INSET)))
  const fy2 = Math.min(MODEL_SIZE - 1, Math.ceil(personBox.y1 + bh * 0.50))

  if ((fx2 - fx1) < 4 || (fy2 - fy1) < 4) return null

  const data = imageData.data
  let faceSum = 0, faceCount = 0
  let varianceAcc = 0

  // First pass: mean luminance of face crop
  for (let py = fy1; py <= fy2; py++) {
    for (let px = fx1; px <= fx2; px++) {
      const idx = (py * MODEL_SIZE + px) * 4
      const lum = data[idx] * 0.299 + data[idx+1] * 0.587 + data[idx+2] * 0.114
      faceSum += lum
      faceCount++
    }
  }
  if (faceCount === 0) return null
  const faceMean = faceSum / faceCount

  // Second pass: variance + brightness histogram within face crop
  let darkPixels = 0, brightPixels = 0
  for (let py = fy1; py <= fy2; py++) {
    for (let px = fx1; px <= fx2; px++) {
      const idx = (py * MODEL_SIZE + px) * 4
      const lum = data[idx] * 0.299 + data[idx+1] * 0.587 + data[idx+2] * 0.114
      const diff = lum - faceMean
      varianceAcc += diff * diff
      if (lum < 50) darkPixels++        // near-black bucket
      if (lum > 200) brightPixels++     // near-white / glare bucket
    }
  }
  const faceVariance  = varianceAcc / faceCount
  const darkFraction  = darkPixels / faceCount
  const brightFraction = brightPixels / faceCount

  // Sample background brightness (area outside person box, full frame)
  let bgSum = 0, bgCount = 0
  for (let py = 0; py < MODEL_SIZE; py += 4) {
    for (let px = 0; px < MODEL_SIZE; px += 4) {
      if (px >= personBox.x1 && px <= personBox.x2 && py >= personBox.y1 && py <= personBox.y2) continue
      const idx = (py * MODEL_SIZE + px) * 4
      const lum = data[idx] * 0.299 + data[idx+1] * 0.587 + data[idx+2] * 0.114
      bgSum += lum
      bgCount++
    }
  }
  const bgMean = bgCount > 0 ? bgSum / bgCount : 0

  //console.log('[facevis] faceMean:', faceMean.toFixed(1), 'variance:', faceVariance.toFixed(1), 'bgMean:', bgMean.toFixed(1))

  const isBacklit  = bgMean > faceMean + 40
  const isDark     = faceMean < 40
  const isFlat     = faceVariance < 180   // silhouette: low internal contrast
  const isBimodal  = (darkFraction > 0.25 && brightFraction > 0.25) ||
                      (darkFraction > 0.5 && brightFraction > 0.03 && faceVariance > 1000)  // glare/halo bleeding into crop, incl. thin rim-light slivers

  if (isDark || (isBacklit && isFlat) || isBimodal) {
    const reason = isBacklit ? 'backlit' : 'low_light'
    return { category: 'FACE_VISIBILITY', severity: 'ADVISORY', confidence: 0.7,
      explanation: reason === 'backlit'
        ? 'Face appears backlit — background much brighter than face'
        : 'Face region too dark to read clearly',
      reason }
  }
  return null
}

const phoneSignal = (dets, smoothed) => {
  const phoneDets = dets.filter(d => {
    if (d.class !== 'cell phone') return false

    // High-confidence YOLO detections are trusted regardless of box shape —
    // aspect ratio alone was rejecting valid phones (e.g. held at an angle,
    // partially occluded by a hand), silently zeroing the signal even
    // though YOLO correctly classified it as "cell phone".
    if (d.score >= CONFIG.HIGH_THRESHOLD) return true

    const bw = d.box.x2 - d.box.x1
    const bh = d.box.y2 - d.box.y1

    const touchesEdge = d.box.x1 < 10 || d.box.y1 < 10 || 
                        d.box.x2 > 630 || d.box.y2 > 630
    if (touchesEdge) return true

    const ratio = bw / bh
    const isPortrait  = ratio > 0.20 && ratio < 0.75
    const isLandscape = ratio > 1.5  && ratio < 2.2
    return isPortrait || isLandscape
  })

  if (phoneDets.length === 0) return 0.0

  const conf = smoothed['cell phone'] ?? 0
  if (conf >= CONFIG.HIGH_THRESHOLD) return 1.0
  if (conf >= CONFIG.LOW_THRESHOLD) return 0.5
  return 0.0
}
const deviceSignal = (dets, smoothed) => {
  const deviceDets = dets.filter(d => 
    d.class === 'laptop' || 
    d.class === 'tv'
  )

  if (deviceDets.length === 0) return 0.0

  // Take highest confidence across all device classes
  const bestConf = Math.max(...deviceDets.map(d => d.score))
  const bestClass = deviceDets.find(d => d.score === bestConf)?.class
  const smoothedConf = smoothed[bestClass] ?? 0

  if (smoothedConf >= CONFIG.HIGH_THRESHOLD) return 1.0
  if (smoothedConf >= CONFIG.LOW_THRESHOLD) return 0.5
  return 0.0
}

// Main monitor
const trackers = [
    new ConditionTracker('PERSON_MISSING', personMissingSignal),
    new ConditionTracker('EXTRA_PERSON', extraPersonSignal),
    new ConditionTracker('PHONE', phoneSignal, 0.85),
    new ConditionTracker('DEVICE', deviceSignal),
];

function getZone(box) {
    const cx = (box.x1 + box.x2) / 2
    const cy = (box.y1 + box.y2) / 2
    
    const zones = []
        if (box.x1 < 80) zones.push("left")
    else if (box.x2 > 560) zones.push("right")
    else if (cx < 160) zones.push("left")
    else if (cx > 480) zones.push("right")
    
    if (box.y1 < 80) zones.push("top")
    else if (box.y2 > 560) zones.push("bottom")
    else if (cy < 160) zones.push("top")
    else if (cy > 480) zones.push("bottom")
    
    if (zones.length === 0) zones.push("center")
    
    return zones.join("-")
}
function getScreenBrightnessState(imageData, box) {
    const x1 = Math.max(0, Math.floor(box.x1));
    const y1 = Math.max(0, Math.floor(box.y1));
    const x2 = Math.min(MODEL_SIZE - 1, Math.ceil(box.x2));
    const y2 = Math.min(MODEL_SIZE - 1, Math.ceil(box.y2));
    const data = imageData.data;

    const bw = x2 - x1;
    const bh = y2 - y1;
    if (bw < 4 || bh < 4) return null;

    // Split box into center region (inner 50%) vs border region (outer ring)
    const cx1 = Math.floor(x1 + bw * 0.25);
    const cy1 = Math.floor(y1 + bh * 0.25);
    const cx2 = Math.floor(x1 + bw * 0.75);
    const cy2 = Math.floor(y1 + bh * 0.75);

    let centerSum = 0, centerCount = 0;
    let borderSum = 0, borderCount = 0;
    let fullSum   = 0, fullCount   = 0;

    for (let py = 0; py < MODEL_SIZE; py++) {
        for (let px = 0; px < MODEL_SIZE; px++) {
            const idx = (py * MODEL_SIZE + px) * 4;
            const lum = data[idx] * 0.299 + data[idx+1] * 0.587 + data[idx+2] * 0.114;
            fullSum += lum; fullCount++;

            if (px >= x1 && px <= x2 && py >= y1 && py <= y2) {
                const inCenter = px >= cx1 && px <= cx2 && py >= cy1 && py <= cy2;
                if (inCenter) { centerSum += lum; centerCount++; }
                else          { borderSum += lum; borderCount++; }
            }
        }
    }

    if (centerCount === 0 || borderCount === 0 || fullCount === 0) return null;

    const centerBrightness = centerSum / centerCount;
    const borderBrightness = borderSum / borderCount;
    const fullBrightness   = fullSum   / fullCount;

    // How much brighter is the box vs the scene
    const sceneRatio = fullBrightness > 0 ? centerBrightness / fullBrightness : 1;
    // How uniformly bright is the box (center vs border)
    // A live screen is uniform — center ≈ border
    // A reflection is patchy — center much brighter than border
    const uniformity = borderBrightness > 0 ? centerBrightness / borderBrightness : 1;

    //console.log('[screen] sceneRatio:', sceneRatio.toFixed(3), 'uniformity:', uniformity.toFixed(3), 'centerBrightness:', centerBrightness.toFixed(1));

    // Screen on: uniform surface AND either brighter than scene OR absolutely bright enough
    if (uniformity < 1.8 && (sceneRatio > 0.75 || centerBrightness > 80)) return 'on';
    // Bright but patchy = reflection
    if (sceneRatio > 0.75 && uniformity >= 1.8) return 'unclear';
    // Dim and patchy = off
    return 'off';
}

// ─── Person lock ─────────────────────────────────────────────
let _lockedPersonBox  = null   // the box we're tracking
let _lockedMissFrames = 0      // consecutive frames without a match
const LOCK_IOU_MIN    = 0.25   // minimum IoU to count as "same person"
const LOCK_MISS_MAX   = 10     // frames before lock resets (person genuinely left)

function getLockedPersonBox(persons) {
  if (persons.length === 0) {
    _lockedMissFrames++
    if (_lockedMissFrames >= LOCK_MISS_MAX) {
      _lockedPersonBox  = null
      _lockedMissFrames = 0
    }
    return _lockedPersonBox
  }

  // If no lock yet, lock onto highest-confidence person
  if (_lockedPersonBox === null) {
    const best = persons.reduce((a, b) => b.score > a.score ? b : a)
    _lockedPersonBox  = best.box
    _lockedMissFrames = 0
    return _lockedPersonBox
  }

  // Find the detection that best overlaps our locked person
  // Reduced from 2% to 0.5% — small/distant persons must still track;
  // below ~32×64 px YOLO can't reliably classify as person anyway.
  const MIN_BOX_AREA = 640 * 640 * 0.005
  let bestIou = 0, bestBox = null
  for (const p of persons) {
    const area = (p.box.x2 - p.box.x1) * (p.box.y2 - p.box.y1)
    if (area < MIN_BOX_AREA) continue
    const overlap = iou(_lockedPersonBox, p.box)
    if (overlap > bestIou) { bestIou = overlap; bestBox = p.box }
  }

  if (bestIou >= LOCK_IOU_MIN) {
    // Scale smoothing by match quality — weak matches barely move the lock,
    // strong matches (person clearly identified) update it more aggressively.
    // IoU 0.25 → a=0.15 (mostly trust old position)
    // IoU 0.75 → a=0.55 (mostly trust new position)
    // IoU 1.00 → a=0.70 (high confidence, follow quickly)
    const a = Math.min(0.70, 0.15 + (bestIou - LOCK_IOU_MIN) * 1.1)
    _lockedPersonBox = {
      x1: _lockedPersonBox.x1 * (1-a) + bestBox.x1 * a,
      y1: _lockedPersonBox.y1 * (1-a) + bestBox.y1 * a,
      x2: _lockedPersonBox.x2 * (1-a) + bestBox.x2 * a,
      y2: _lockedPersonBox.y2 * (1-a) + bestBox.y2 * a,
    }
    _lockedMissFrames = 0
  } else {
    // No detection overlaps our person — they may have moved or left
    _lockedMissFrames++
    if (_lockedMissFrames >= LOCK_MISS_MAX) {
      _lockedPersonBox  = null
      _lockedMissFrames = 0
    }
    // Return stale box for now so motion worker doesn't lose exclusion zone
  }

  return _lockedPersonBox
}

function onDetectionFrame(detections, imageData) {
    const now = performance.now();
    const results = trackers.map(t => ({
        condition: t.name,
        state: t.update(detections, now),
        evidence: t.evidence,
    }));

    const phoneRawConf = detections
    .filter(d => d.class === 'cell phone')
    .reduce((max, d) => Math.max(max, d.score), 0)
   // if (phoneRawConf > 0) console.log('[DEBUG onDetectionFrame]', phoneRawConf, 'rawDets:', detections.filter(d => d.class === 'cell phone').map(d => d.score))

const phoneDet = detections.find(d => d.class === 'cell phone')
    const phoneZone = phoneDet ? getZone(phoneDet.box) : null

    const persons = detections.filter(d => d.class === 'person' && d.score > 0.5)
    const personZones = persons.map(d => getZone(d.box))

    const personBox = getLockedPersonBox(persons)

    const overall = results.some(r => r.state === 'RED') ? 'RED'
        : results.some(r => r.state === 'YELLOW') ? 'YELLOW'
            : 'OK';

    const rawPersonCount = detections.filter(d => d.class === 'person' && d.score > 0.5).length

    const deviceDets = detections.filter(d => d.class === 'laptop' || d.class === 'tv')
    const deviceRawConf = deviceDets.reduce((max, d) => Math.max(max, d.score), 0)
    const bestDeviceDet = deviceDets.reduce((best, d) => (!best || d.score > best.score) ? d : best, null)
    const deviceZone = bestDeviceDet ? getZone(bestDeviceDet.box) : null
    const deviceClass = bestDeviceDet?.class || null
    let phoneScreenState = null;
    if (phoneDet) {
        phoneScreenState = getScreenBrightnessState(imageData, phoneDet.box);
    }
    const phoneIncident = classifyPhone(detections, imageData)
    const deviceIncident = classifyDevice(detections, imageData)

    faceVisProcessedCount++
    let faceVisibilityIncident = null
    if (faceVisProcessedCount > FACE_VIS_WARMUP_FRAMES) {
      faceVisibilityIncident = classifyFaceVisibility(imageData, personBox)
    }

    return { results, overall, type: "update", phoneRawConf, phoneZone, personZones, personBox, rawPersonCount, deviceRawConf, deviceZone, deviceClass, phoneScreenState, phoneIncident, deviceIncident, faceVisibilityIncident, faceVisible: faceVisibilityIncident === null && personBox !== null }
}

async function startProcessing() {
    if (processing) {
        return;
    }
    processing = true;
    const intervalRef = setInterval(() => {

        // console.debug("[PIPELINE]", {

        //     receivedFPS:
        //         (metrics.received / 5).toFixed(1),

        //     processedFPS:
        //         (metrics.processed / 5).toFixed(1),

        //     overwrittenFPS:
        //         (metrics.overwritten / 5).toFixed(1),

        //     preprocessMs:
        //         avg(metrics.preprocessMs).toFixed(1),

        //     inferenceMs:
        //         avg(metrics.inferenceMs).toFixed(1),

        //     decodeMs:
        //         avg(metrics.decodeMs).toFixed(1),

        //     totalMs:
        //         avg(metrics.totalMs).toFixed(1),

        //     frameAgeMs:
        //         avg(metrics.frameAgeMs).toFixed(1)

        // });

        resetMetrics();

    }, 5000);

    while (true) {
        if (!session) {
            await sleep(100);
            continue;
        }

        if (!latestFrame) {
            await sleep(5);
            continue;
        }
        const frame = latestFrame;

        latestFrame = null;

        try {
            processedFrames++;
            metrics.processed++;

            const start = performance.now();

            const ageMs = (start - (frame.timestamp / 1000)).toFixed(1);

            //console.debug("[worker processing]", { timestamp: frame.timestamp, ageMs });

            const t0 = performance.now();

            const { tensor, meta, snapshotBitmap, imageData } = await preprocess(frame);

            const t1 = performance.now();

            safeClose(frame);

            const results = await session.run({ images: tensor });

            const t2 = performance.now();

            const output = results[Object.keys(results)[0]];
            const detections = decodeYOLO26(output, labels);
            const t3 = performance.now();
            metrics.preprocessMs.push(t1 - t0);
            metrics.inferenceMs.push(t2 - t1);
            metrics.decodeMs.push(t3 - t2);
            metrics.totalMs.push(t3 - t0);

            metrics.frameAgeMs.push(ageMs);

            
            snapshotBitmap.close();
            // OPTIONAL LOG
            const update = onDetectionFrame(detections, imageData);

// console.log(
//   "[DEBUG before postMessage]",
//   update.phoneRawConf
// );

postMessage(update);


        } catch (err) {

            postMessage({
                type: "error",
                error: err.message
            });

        }

    }
    processing = false
    clearInterval(intervalRef);



}

startProcessing().catch((err) => console.error(err));