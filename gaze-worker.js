'use strict';

self.addEventListener('error', (e) => {
  console.error('gaze-worker internal error:', e.message, e.filename, e.lineno);
});

importScripts('./mediapipe.js');

const { FaceLandmarker, FilesetResolver } = self.$mediapipe;

// ─── Thresholds ──────────────────────────────────────────────
const CALIBRATION_FRAMES  = 30          // frames to collect baseline
const SOFT_THRESHOLD      = 8         // deviation → glancing
const HARD_THRESHOLD      = 14         // deviation → away
const GLANCE_GRACE_MS     = 2000        // must be away this long → away state
const ABSENT_GRACE_MS     = 1000        // face missing this long → absent
const FREQUENCY_WINDOW_MS = 60000       // 60s rolling window
const OSCILLATION_WINDOW_MS = 30000     // 30s rolling window for rapid switches
const AWAY_LIMIT_60       = 20          // seconds away in 60s → suspicious
const EVENTS_LIMIT_60     = 4           // away events in 60s → suspicious
const OSCILLATION_LIMIT   = 6          // direction switches in 30s → suspicious

// ─── State ───────────────────────────────────────────────────
let landmarker        = null

// calibration
let calibrating       = true
let calibFrames       = []              // array of {pitch, yaw, roll}
let baseline          = null           // {pitch, yaw, roll}
let baselineMAD       = null           // median absolute deviation per axis

// silent eye baseline (collected after head calibration)
let eyeCalibrating    = false
let eyeCalibFrames    = []
let eyeBaseline       = null           // {x, y}

// eye alert tracking
let eyeAlertEvents    = []             // timestamps of significant eye movements
let lastEyeAlertSent  = 0             // last time we sent a 60s summary
let eyeGazeStart      = null          // when current sustained corner-gaze episode began
let lastEyeGazeCountAt = null         // timestamp of the last tick counted within this episode
const EYE_ALERT_WINDOW_MS    = 60000
const EYE_ALERT_THRESHOLD    = 5
const EYE_MAGNITUDE_THRESHOLD = 0.22
const EYE_SUSTAIN_MS         = 1500

// state machine
let currentState      = 'focused'      // focused | glancing | away | absent | unstable
let awayStartTime     = null
let absentStartTime   = null
let glanceStartTime   = null

// rolling windows
let awayEvents        = []             // timestamps of away events (last 60s)
let awayDurations     = []             // {start, end} of away periods (last 60s)
let directionSwitches = []             // timestamps of direction switches (last 30s)
let lastDirection     = null           // 'left'|'right'|'up'|'down'|null

let irisSmoothing = []
const IRIS_SMOOTH_FRAMES = 6
const GAZE_WARMUP_FRAMES = 20
let gazeFrameCount = 0
let lastFaceVisible = true   // optimistic default; updated from worker.js signal


// ─── Math helpers ────────────────────────────────────────────
function median(arr) {
  const s = [...arr].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m-1] + s[m]) / 2
}

function mad(arr, med) {
  return median(arr.map(v => Math.abs(v - med)))
}

function computeDeviation(pitch, yaw, roll, eyeGaze) {
  if (!baseline) return 0
  const dp = Math.abs(pitch - baseline.pitch)
  const dy = Math.abs(yaw   - baseline.yaw)
  const dr = Math.abs(roll  - baseline.roll)
  const base = dy * 0.5 + dp * 0.35 + dr * 0.15

  if (!eyeGaze) return base

  // Head direction deltas
  const dYaw   = yaw   - baseline.yaw
  const dPitch = pitch - baseline.pitch

  const headRight = dYaw   >  3
  const headLeft  = dYaw   < -3
  const headDown  = dPitch >  3
  const headUp    = dPitch < -3

  const eyeRight = eyeGaze.x >  0.2
  const eyeLeft  = eyeGaze.x < -0.2
  const eyeDown  = eyeGaze.y >  0.2
  const eyeUp    = eyeGaze.y < -0.2

  const corroborated =
    (eyeRight && headRight) ||
    (eyeLeft  && headLeft)  ||
    (eyeDown  && headDown)  ||
    (eyeUp    && headUp)

  if (corroborated) return base * 1.8

  // Eyes alone — meaningful independent contribution (50% weight)
  const eyeMagnitude = Math.sqrt(eyeGaze.x * eyeGaze.x + eyeGaze.y * eyeGaze.y)
  const eyeContribution = eyeMagnitude * 15 * 0.5

  return base + eyeContribution
}

function computeEyeGaze(landmarks) {
  if (!landmarks) return null

  const leftIris  = landmarks[468]
  const rightIris = landmarks[473]

  // Eye socket corners
  const leftOuter  = landmarks[33]
  const leftInner  = landmarks[133]
  const rightOuter = landmarks[263]
  const rightInner = landmarks[362]

  // Eye top/bottom for vertical
  const leftTop    = landmarks[159]
  const leftBottom = landmarks[145]
  const rightTop   = landmarks[386]
  const rightBottom = landmarks[374]

  // Horizontal: where iris sits within eye socket (0=outer edge, 1=inner edge)
  // Normalize to -1 to 1 (0 = center)
  const leftSpanX  = Math.abs(leftInner.x  - leftOuter.x)
  const rightSpanX = Math.abs(rightInner.x - rightOuter.x)
  const leftGazeX  = leftSpanX  > 0 ? (leftIris.x  - leftOuter.x)  / leftSpanX  : 0.5
  const rightGazeX = rightSpanX > 0 ? (rightIris.x - rightOuter.x) / rightSpanX : 0.5
  const gazeX = ((leftGazeX + rightGazeX) / 2) * 2 - 1

  // Vertical
  const leftSpanY  = Math.abs(leftBottom.y  - leftTop.y)
  const rightSpanY = Math.abs(rightBottom.y - rightTop.y)
  const leftGazeY  = leftSpanY  > 0 ? (leftIris.y  - leftTop.y)  / leftSpanY  : 0.5
  const rightGazeY = rightSpanY > 0 ? (rightIris.y - rightTop.y) / rightSpanY : 0.5
  const gazeY = ((leftGazeY + rightGazeY) / 2) * 2 - 1

  irisSmoothing.push({ x: gazeX, y: gazeY })
if (irisSmoothing.length > IRIS_SMOOTH_FRAMES) irisSmoothing.shift()
const avgX = irisSmoothing.reduce((s, f) => s + f.x, 0) / irisSmoothing.length
const avgY = irisSmoothing.reduce((s, f) => s + f.y, 0) / irisSmoothing.length
const leftOpenness  = Math.abs(leftBottom.y - leftTop.y) / (leftSpanY || 0.01)
  const rightOpenness = Math.abs(rightBottom.y - rightTop.y) / (rightSpanY || 0.01)
  if ((leftOpenness + rightOpenness) / 2 < 0.15) return null

  return { x: avgX, y: avgY }
}


function checkEyeAlerts(calibratedEyeGaze, now) {
  if (!calibratedEyeGaze) return
  if (calibratedEyeGaze) {
    const mag = Math.sqrt(calibratedEyeGaze.x ** 2 + calibratedEyeGaze.y ** 2)
    //if (mag > 0.1) console.log('[eye] x:', calibratedEyeGaze.x.toFixed(3), 'y:', calibratedEyeGaze.y.toFixed(3), 'mag:', mag.toFixed(3), '| events in window:', eyeAlertEvents.length, '/ need:', EYE_ALERT_THRESHOLD)
  }
  if (currentState !== 'focused' && currentState !== 'glancing') return

  // Require both eyes to agree — magnitude check on combined signal
  const magnitude = Math.sqrt(
    calibratedEyeGaze.x * calibratedEyeGaze.x +
    calibratedEyeGaze.y * calibratedEyeGaze.y
  )

  if (magnitude > EYE_MAGNITUDE_THRESHOLD) {
    if (eyeGazeStart === null) {
      eyeGazeStart = now
      lastEyeGazeCountAt = now
    }
    // Re-arms every EYE_SUSTAIN_MS within the same continuous episode,
    // so a long stare keeps accumulating events instead of counting once.
    if (now - lastEyeGazeCountAt >= EYE_SUSTAIN_MS) {
      eyeAlertEvents.push(now)
      lastEyeGazeCountAt = now
      console.log('[eye] sustained gaze tick counted — episode age:', Math.round(now - eyeGazeStart), 'ms | total events:', eyeAlertEvents.length, '/ need:', EYE_ALERT_THRESHOLD)
    }
  } else {
    // Eyes returned to center — reset episode
    eyeGazeStart = null
    lastEyeGazeCountAt = null
  }

  // Prune to rolling 60s window — events older than 60s fall off naturally
  eyeAlertEvents = eyeAlertEvents.filter(t => t > now - EYE_ALERT_WINDOW_MS)

  const count = eyeAlertEvents.length

  // Fire as soon as threshold is crossed, with a cooldown so it doesn't
  // re-fire every single frame while still above threshold
  const ALERT_COOLDOWN_MS = 30000
  if (count >= EYE_ALERT_THRESHOLD && (now - lastEyeAlertSent >= ALERT_COOLDOWN_MS)) {
    lastEyeAlertSent = now

    const severity = count >= 10 ? 'high' : count >= 8 ? 'medium' : 'low'
    const message = count >= 10
      ? `Candidate's eyes were highly active while appearing focused`
      : count >= 8
      ? `Candidate showed excessive eye movement — possible screen scanning`
      : `Candidate showed frequent eye movement while focused`

    self.postMessage({
      type: 'eyeAlert',
      count,
      severity,
      message
    })
  }
}

function getDirection(pitch, yaw) {
  if (!baseline) return null
  const dp = pitch - baseline.pitch
  const dy = yaw   - baseline.yaw
  if (Math.abs(dy) > Math.abs(dp)) return dy > 0 ? 'right' : 'left'
  return dp > 0 ? 'down' : 'up'
}

// ─── Calibration ─────────────────────────────────────────────
function addCalibrationFrame(pitch, yaw, roll) {
  calibFrames.push({ pitch, yaw, roll })
  if (calibFrames.length < CALIBRATION_FRAMES) return false

  baseline = {
    pitch: median(calibFrames.map(f => f.pitch)),
    yaw:   median(calibFrames.map(f => f.yaw)),
    roll:  median(calibFrames.map(f => f.roll)),
  }
  baselineMAD = {
    pitch: mad(calibFrames.map(f => f.pitch), baseline.pitch),
    yaw:   mad(calibFrames.map(f => f.yaw),   baseline.yaw),
    roll:  mad(calibFrames.map(f => f.roll),   baseline.roll),
  }
calibrating = false
  eyeCalibrating = true
  //console.log('[gaze-worker] baseline ready', baseline, baselineMAD)
  //console.log('[gaze-worker] starting silent eye calibration...')
  return true
}

// ─── Rolling window helpers ──────────────────────────────────
function pruneWindow(arr, windowMs) {
  const cutoff = Date.now() - windowMs
  return arr.filter(t => t > cutoff)
}

function getTotalAwaySeconds() {
  const now = Date.now()
  const cutoff = now - FREQUENCY_WINDOW_MS
  return awayDurations
    .filter(d => d.end > cutoff)
    .reduce((sum, d) => sum + (Math.min(d.end, now) - Math.max(d.start, cutoff)) / 1000, 0)
}

// ─── Score computation ───────────────────────────────────────
function computeScore() {
  let score = 0

  const awayEvents60  = awayEvents.length
  const awayDur60     = getTotalAwaySeconds()
  const oscillations  = directionSwitches.length

  if (currentState === 'away')    score += 40
  if (currentState === 'absent')  score += 50
  if (currentState === 'unstable') score += 30
  if (currentState === 'glancing') score += 10

  if (awayDur60 > AWAY_LIMIT_60)        score += 25
  if (awayEvents60 >= EVENTS_LIMIT_60)  score += 20
  if (oscillations >= OSCILLATION_LIMIT) score += 15

  return Math.min(score, 100)
}

// ─── State machine ───────────────────────────────────────────
function updateState(deviation, facePresent, now) {
  if (!facePresent) {
    if (absentStartTime === null) absentStartTime = now
    if (now - absentStartTime >= ABSENT_GRACE_MS) {
      currentState = 'absent'
    }
    awayStartTime   = null
    glanceStartTime = null
    return
  }

  absentStartTime = null

  if (deviation < SOFT_THRESHOLD) {
    // focused
    if (currentState === 'away' && awayStartTime !== null) {
      // record completed away period
      awayDurations.push({ start: awayStartTime, end: now })
      awayDurations = awayDurations.filter(d => d.end > now - FREQUENCY_WINDOW_MS)
    }
    awayStartTime   = null
    glanceStartTime = null
    currentState    = 'focused'
  } else if (deviation < HARD_THRESHOLD) {
    // glancing
    if (glanceStartTime === null) glanceStartTime = now
    currentState = 'glancing'
    awayStartTime = null
  } else {
    // away territory
    if (awayStartTime === null) {
      awayStartTime   = now
      glanceStartTime = null
    }
    if (now - awayStartTime >= GLANCE_GRACE_MS) {
      if (currentState !== 'away') {
        // just entered away state — log event
        awayEvents.push(now)
        awayEvents = pruneWindow(awayEvents, FREQUENCY_WINDOW_MS)
      }
      currentState = 'away'
    } else {
      currentState = 'glancing'
    }
  }
}

function trackOscillation(pitch, yaw) {
  if (!baseline) return
  const deviation = computeDeviation(pitch, yaw, 0, null)
  if (deviation < SOFT_THRESHOLD) {
    lastDirection = null
    return
  }
  const dir = getDirection(pitch, yaw)
  if (dir && dir !== lastDirection) {
    directionSwitches.push(Date.now())
    directionSwitches = pruneWindow(directionSwitches, OSCILLATION_WINDOW_MS)
    lastDirection = dir
  }
}

// ─── Init ────────────────────────────────────────────────────
async function resolveAsset(localPath, cdnUrl) {
  try {
    const res = await fetch(localPath, { method: 'HEAD' })
    if (res.ok) {
      console.log('[gaze-worker] using local asset:', localPath)
      return localPath
    }
  } catch {}
  console.log('[gaze-worker] local asset not found, falling back to CDN:', cdnUrl)
  return cdnUrl
}

async function init() {
  const wasmPath = await resolveAsset(
    './mediapipe-local/wasm/vision_wasm_internal.js',
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm'
  )
  // forVisionTasks expects the folder, strip filename if local
  const wasmFolder = wasmPath.endsWith('.js')
    ? wasmPath.replace('/vision_wasm_internal.js', '')
    : wasmPath

  const modelPath = await resolveAsset(
    './mediapipe-local/face_landmarker.task',
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
  )

  const vision = await FilesetResolver.forVisionTasks(wasmFolder)
  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: modelPath,
      delegate: 'GPU'
    },
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: true,
    runningMode: 'VIDEO',
    numFaces: 1
  })
  self.postMessage({ type: 'ready' })
}

// ─── Message handler ─────────────────────────────────────────
self.onmessage = async (e) => {
  const msg = e.data

  if (msg.type === 'recalibrate') {
    calibrating    = true
    calibFrames    = []
    baseline       = null
    baselineMAD    = null
    eyeCalibrating = false
    eyeCalibFrames = []
    eyeBaseline    = null
    currentState   = 'focused'
    awayStartTime  = null
    absentStartTime = null
    glanceStartTime = null
    awayEvents     = []
    awayDurations  = []
    directionSwitches = []
    lastDirection  = null
    gazeFrameCount = 0
    lastFaceVisible = true
    console.log('[gaze-worker] recalibrating...')
    return
  }

if (msg.type !== 'frame') return
  if (!landmarker) { msg.frame.close(); return }

  gazeFrameCount++
  if (msg.faceVisible !== undefined) lastFaceVisible = msg.faceVisible
  if (gazeFrameCount <= GAZE_WARMUP_FRAMES) { msg.frame.close(); return }

  const result = landmarker.detectForVideo(msg.frame, msg.timestamp)
  const personBox = msg.personBox ?? null
  msg.frame.close()

  const now = Date.now()

  if (!result.facialTransformationMatrixes?.length) {
    updateState(0, false, now)
    self.postMessage({
      type: 'gaze',
      state: currentState,
      score: computeScore(),
      awayEventCount: awayEvents.length,
      awayDuration60: Math.round(getTotalAwaySeconds()),
      oscillations30: directionSwitches.length,
      baselineReady: !calibrating,
      pitch: 0, yaw: 0
    })
    return
  }

const matrix = result.facialTransformationMatrixes[0].data
const pitch  = Math.asin(-matrix[6])  * (180 / Math.PI)
const yaw    = Math.atan2(matrix[2], matrix[10]) * (180 / Math.PI)
const roll   = Math.atan2(matrix[4], matrix[5])  * (180 / Math.PI)

// Nose tip (landmark 4) gives stable face centroid, scale-invariant
const landmarks = result.faceLandmarks?.[0]
const faceX = landmarks ? landmarks[4].x : null
const faceY = landmarks ? landmarks[4].y : null

// If we have a locked person box, reject faces that are clearly outside it.
// personBox coords are in YOLO model space (0–640); face landmarks are 0–1 normalized.
// Only apply this check once baseline is established (calibrating phase is fine to skip).
if (personBox && landmarks && !calibrating) {
  const faceCx = faceX  // already 0–1
  const faceCy = faceY
  // Convert personBox to 0–1
  const pbx1 = personBox.x1 / 640, pbx2 = personBox.x2 / 640
  const pby1 = personBox.y1 / 640, pby2 = personBox.y2 / 640
  // Expand box by 20% to allow for detection jitter
  const pad = 0.20
  const bw = pbx2 - pbx1, bh = pby2 - pby1
  const inBox = faceCx >= pbx1 - bw * pad && faceCx <= pbx2 + bw * pad &&
                faceCy >= pby1 - bh * pad && faceCy <= pby2 + bh * pad
  if (!inBox) {
    // Face detected but it's not our locked person — ignore this frame
    return
  }
}

  // calibration phase
  if (calibrating) {
    if (!lastFaceVisible) {
      // Face not legible — stall calibration, don't consume garbage frames
      self.postMessage({
        type: 'gaze',
        state: 'calibrating',
        score: 0,
        awayEventCount: 0,
        awayDuration60: 0,
        oscillations30: 0,
        baselineReady: false,
        calibrationProgress: Math.round((calibFrames.length / CALIBRATION_FRAMES) * 100),
        pitch, yaw
      })
      return
    }
    const done = addCalibrationFrame(pitch, yaw, roll)
    self.postMessage({
      type: 'gaze',
      state: 'calibrating',
      score: 0,
      awayEventCount: 0,
      awayDuration60: 0,
      oscillations30: 0,
      baselineReady: false,
      calibrationProgress: Math.round((calibFrames.length / CALIBRATION_FRAMES) * 100),
      pitch, yaw
    })
    return
  }

const eyeGaze = computeEyeGaze(landmarks)
const now2 = Date.now()

  // Silent eye baseline collection — only accept frames where head is stable
  if (eyeCalibrating && eyeGaze) {
    const headDeviation = computeDeviation(pitch, yaw, roll, null)
    if (headDeviation < SOFT_THRESHOLD) {
      eyeCalibFrames.push({ x: eyeGaze.x, y: eyeGaze.y })
      if (eyeCalibFrames.length >= 20) {
        eyeBaseline = {
          x: median(eyeCalibFrames.map(f => f.x)),
          y: median(eyeCalibFrames.map(f => f.y))
        }
        eyeCalibrating = false
        console.log('[gaze-worker] eye baseline locked', eyeBaseline)
      }
    }
  }

  // Offset eye gaze from personal baseline before passing to deviation
  const calibratedEyeGaze = (eyeGaze && eyeBaseline) ? {
    x: eyeGaze.x - eyeBaseline.x,
    y: eyeGaze.y - eyeBaseline.y
  } : null

  const deviation = computeDeviation(pitch, yaw, roll, calibratedEyeGaze)
  updateState(deviation, true, now)
  trackOscillation(pitch, yaw)
  checkEyeAlerts(calibratedEyeGaze, now2)
  const dYaw = baseline ? yaw - baseline.yaw : 0
  const dPitch = baseline ? pitch - baseline.pitch : 0

  self.postMessage({
    type: 'gaze',
    state: currentState,
    score: computeScore(),
    awayEventCount: awayEvents.length,
    awayDuration60: Math.round(getTotalAwaySeconds()),
    oscillations30: directionSwitches.length,
    baselineReady: true,
    deviation: Math.round(deviation),
    pitch: Math.round(pitch),
    yaw:   Math.round(yaw),
    roll: Math.round(roll),
    eyeX: calibratedEyeGaze ? Math.round(calibratedEyeGaze.x * 100) / 100 : null,
eyeY: calibratedEyeGaze ? Math.round(calibratedEyeGaze.y * 100) / 100 : null,
    faceX,
    faceY,
    dYaw: Math.round(dYaw),
    dPitch: Math.round(dPitch)
  })
}

init().catch(e => {
  console.error('gaze-worker init failed:', e)
  self.postMessage({ type: 'error', message: e?.message || String(e) })
})