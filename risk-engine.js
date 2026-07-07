let signals = {
  framesAnalyzed: 0,
  faceAbsentCount: 0,
  multipleFacesCount: 0,
  gazeAwayCount: 0,
  phoneDetectedCount: 0,
  deviceDetectedCount: 0
}

let currentRisk = 0
let displayRisk = 0;
let samplingInterval = 2000
let lastRiskUpdateTime = Date.now()

const RISK_ACCUMULATE_RATE = 0.35       // fraction of windowScore pulled in per SCORE_INTERVAL_MS
const RISK_DECAY_HALF_LIFE_MS = 3000    // risk halves every 3s once signal clears
let activeIncidentId = null

// ─── Candidate → proctor relay ─────────────────────────────────
// All former DOM-rendering calls (setBeliefBar, setPill, createIncidentCard,
// addFlag, removeFlag, closeIncidentCard, updateRiskTimeline, etc.) are
// redirected here instead of touching the DOM. window.sendToProctor is
// wired up by candidate.html once the WebSocket connection is ready; it
// already handles queuing/buffering if the socket isn't open yet, so this
// module doesn't need to know about connection state at all.
function sendToProctor(type, payload) {
  if (typeof window.sendToProctor === 'function') {
    // payload may itself carry a "type" field (e.g. gaze-worker's raw
    // eyeAlert message has type: 'eyeAlert') — strip it so it can't
    // clobber the outer wire-protocol type we're setting here.
    const { type: _ignored, ...rest } = payload || {}
    window.sendToProctor({ type, ...rest })
  }
}

// A single shared canvas used to convert ImageBitmap proof frames into a
// base64 JPEG string before they go over the wire — ImageBitmap objects
// cannot be structured-cloned through a WebSocket. Every code path that
// attaches a "proof" field to an outgoing message must route through
// bitmapToDataURL so the conversion happens exactly one way, consistently.
let _proofCanvas = null
let _proofCtx = null
function bitmapToDataURL(bitmap) {
  if (!bitmap) return null
  if (!_proofCanvas) {
    _proofCanvas = document.createElement('canvas')
    _proofCtx = _proofCanvas.getContext('2d')
  }
  _proofCanvas.width = bitmap.width
  _proofCanvas.height = bitmap.height
  _proofCtx.drawImage(bitmap, 0, 0)
  const dataUrl = _proofCanvas.toDataURL('image/jpeg', 0.7)
  // The bitmap has now been drawn to canvas; the caller is responsible for
  // closing it if it owns the bitmap's lifecycle (mirrors existing
  // msg.proof.close() behavior from the old addFlag()).
  return dataUrl
}

// Converts every ImageBitmap found on well-known proof-bearing fields of an
// incident-like object into a base64 string, returning a shallow copy safe
// to JSON.stringify and send over the WebSocket.
function serializeIncidentForWire(incident) {
  if (!incident) return incident
  const out = { ...incident }
  if (out.proof && typeof out.proof.close === 'function') {
    out.proof = bitmapToDataURL(out.proof)
  }
  return out
}

let consecutiveHighTicks = 0
let consecutiveLowTicks = 0
let lastIncidentCloseTime = 0
const CONFIRMATION_TICKS = 3
const INCIDENT_COOLDOWN_MS = 10000
let incidentStartTime = null
let suspicionStartTime = null
let lastSuspiciousTime = null
let lastPhoneZone = null
let lastPersonZones = []
let lastDeviceZone = null
let lastDeviceClass = null
let conditionDurations = {
  phoneVisible: 0,
  personAbsent: 0,
  multiPerson: 0
}

// ─── Face centroid position tracking ─────────────────────────
let faceCentroidHistory = []
let homePosition = null
const HOME_CALIBRATION_FRAMES = 30
const DRIFT_THRESHOLD = 0.15
const DRIFT_SUSTAIN_MS = 8000
const POSITION_FLAG_COOLDOWN = 90000
const POSITION_RESOLVE_GRACE = 8000
let driftStartTime = null
let activePositionFlag = false
let lastPositionFlagId = null
let lastPositionFlagTime = null
let positionClearStartedAt = null

let lastBehavioralFlagTime = null
let behavioralClearStartedAt = null
const BEHAVIORAL_COOLDOWN = 60000
const BEHAVIORAL_RESOLVE_GRACE = 8000
const SCORE_INTERVAL_MS = 5000
let activeBehavioralFlag = false
let lastBehavioralFlagId = null

// ─── Eye tracking ─────────────────────────────────────────────
let eyeTrackingEnabled = true
export function setEyeTracking(enabled) { eyeTrackingEnabled = enabled }

export function updateEyeAlerts(msg) {
  //console.log('updateEyeAlerts called, eyeTrackingEnabled =', eyeTrackingEnabled, msg)
  if (!eyeTrackingEnabled) return

  // msg comes straight from gaze-worker's 'eyeAlert' postMessage:
  // { type: 'eyeAlert', count, severity, message }
  sendToProctor('eye_alert', msg)
}

// ─── Face visibility advisory ────────────────────────────────
// Tracks last-sent state so we only notify the proctor on actual
// transitions, matching the old toast's show-once/hide behavior instead
// of firing on every frame.
let _lastFaceVisNotice = null // { visible, reason } | null

export function showFaceVisibilityNotice(visible, reason) {
  // Face-not-visible notification disabled — intentionally a no-op.
  return
}

// ─── Gaze flag state ─────────────────────────────────────────
let lastGazeMsg = null              // most recent gaze worker message
let gazeAwayStreakStart = null
const activeIncidents = {} // category -> { id, severity }
let activeGazeFlag = false
let lastGazeFlagId = null
let lastGazeFlagTime = null
let gazeClearStartedAt = null
let gazeScoreAccum = 0             // rolling accumulator for direct gaze score injection
let gazeScoreSamples = 0
const GAZE_FLAG_COOLDOWN = 30000   // 30s between repeated gaze-only flags
const GAZE_RESOLVE_GRACE = 10000   // 10s before resolving a gaze flag

const CATEGORY_CONFIRM_TICKS = 8
const CATEGORY_COOLDOWN_MS = 60000
const categoryTicks = {}
const lastCategoryCloseTime = {}

function evaluateIncident(category, incident) {
  //console.log("evaluateIncident", category, incident)
  const now = Date.now()
  const t = categoryTicks[category] || (categoryTicks[category] = { hitTicks: 0, clearTicks: 0 })

  if (incident) {
    t.hitTicks++
    t.clearTicks = 0
    const cooldownPassed = now - (lastCategoryCloseTime[category] || 0) > CATEGORY_COOLDOWN_MS
    const alreadyActive = !!activeIncidents[category]
    if (!alreadyActive && t.hitTicks >= CATEGORY_CONFIRM_TICKS && cooldownPassed) {
      triggerIncident(category, incident)
    }
  } else {
    t.clearTicks++
    // Leaky bucket: raw per-frame detections (phone/multi-person) flicker
    // frame to frame far more than a sustained gaze-away streak does, so a
    // single missed frame used to wipe hitTicks to 0 and the "8 in a row"
    // requirement was almost never met for those signals even when the
    // condition was genuinely present most of the time. Only start
    // bleeding progress once a miss repeats; resolving an already-active
    // incident still works exactly as before.
    if (t.clearTicks > 1) {
      t.hitTicks = Math.max(0, t.hitTicks - 1)
    }
    if (activeIncidents[category] && t.clearTicks >= CATEGORY_CONFIRM_TICKS) {
      resolveIncident(category)
      lastCategoryCloseTime[category] = now
    }
  }
}

export function scoreWindow() {
  const total = signals.framesAnalyzed
  if (total === 0) return 0

  let score = 0

  const absentRatio = signals.faceAbsentCount / total
  if (absentRatio > 0.5) score += 40
  else if (absentRatio > 0.2) score += 20

  if (signals.multipleFacesCount > 2) score += 35

  const phoneRatio = signals.phoneDetectedCount / total
  if (phoneRatio > 0.3) score += 40
  else if (phoneRatio > 0.1) score += 20

const gazeRatio = signals.gazeAwayCount / total
  if (gazeRatio > 0.4) score += 40
  else if (gazeRatio > 0.2) score += 20

  const deviceRatio = signals.deviceDetectedCount / total
  if (deviceRatio > 0.3) score += 35
  else if (deviceRatio > 0.1) score += 15

  // Direct gaze score contribution (weighted 40% of raw gaze score, capped at 30pts)
  // This captures nuanced gaze patterns (oscillations, duration) beyond binary away/not-away
  if (gazeScoreSamples > 0) {
    const avgGazeScore = gazeScoreAccum / gazeScoreSamples
    const gazeContribution = Math.min(30, Math.round(avgGazeScore * 0.4))
    score += gazeContribution
  }

  return Math.min(score, 100)
}

export function buildReasons() {
  const reasons = []
  const total = signals.framesAnalyzed
  if (total === 0) return reasons
  const absentRatio = signals.faceAbsentCount / total
  if (absentRatio > 0.5)
    reasons.push(`Face absent in ${Math.round(absentRatio * 100)}% of frames`)
  if (signals.multipleFacesCount > 2)
    reasons.push(`Multiple faces detected ${signals.multipleFacesCount} times`)
  const gazeRatio = signals.gazeAwayCount / total
  if (gazeRatio > 0.4)
    reasons.push(`Gaze away ${Math.round(gazeRatio * 100)}% of time`)
  return reasons
}

export function updateSamplingRate() {
  if (currentRisk < 50) samplingInterval = 2000
  else if (currentRisk < 70) samplingInterval = 1000
  else if (currentRisk < 90) samplingInterval = 500
  else samplingInterval = 200
}

function medianVal(arr) {
  const s = [...arr].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

function getDriftDistance(x, y) {
  if (!homePosition) return 0
  const dx = x - homePosition.x
  return Math.abs(dx)
}

function getDriftDirection(x, y) {
  if (!homePosition) return null
  const dx = x - homePosition.x
  const dy = y - homePosition.y
  if (Math.abs(dx) >= Math.abs(dy)) return dx > 0 ? 'right' : 'left'
  return dy > 0 ? 'down' : 'up'
}

function updateFacePosition(faceX, faceY) {
  if (faceX === null || faceY === null) return

  // Build home position from first N frames
  if (!homePosition) {
    faceCentroidHistory.push({ x: faceX, y: faceY })
    if (faceCentroidHistory.length >= HOME_CALIBRATION_FRAMES) {
      homePosition = {
        x: medianVal(faceCentroidHistory.map(p => p.x)),
        y: medianVal(faceCentroidHistory.map(p => p.y))
      }
      //console.log('[position] home position set', homePosition)
    }
    return
  }

  const now = Date.now()
  const drift = getDriftDistance(faceX, faceY)
  const cooldownPassed = !lastPositionFlagTime || (now - lastPositionFlagTime > POSITION_FLAG_COOLDOWN)

  if (drift > DRIFT_THRESHOLD) {
    if (!driftStartTime) driftStartTime = now
    positionClearStartedAt = null

    if (!activePositionFlag && cooldownPassed && (now - driftStartTime >= DRIFT_SUSTAIN_MS)) {
      const direction = getDriftDirection(faceX, faceY)
      const pct = Math.round(drift * 100)
      triggerPositionFlag({
        type: 'BEHAVIORAL',
        emoji: '🔵',
        severity: 'LOW',
        description: `Candidate shifted ${direction} from seated position — ${pct}% drift sustained for ${Math.round((now - driftStartTime) / 1000)}s`,
        correlated: false
      })
    }
  } else {
    driftStartTime = null
    if (activePositionFlag) {
      if (!positionClearStartedAt) positionClearStartedAt = now
      if (now - positionClearStartedAt >= POSITION_RESOLVE_GRACE) {
        resolvePositionFlag()
      }
    }
  }
}

function classifyGazeAway(gaze) {
  const now = Date.now()
  const isAway = gaze.state === 'away' || gaze.state === 'absent'
  if (!isAway) { gazeAwayStreakStart = null; return null }
  if (!gazeAwayStreakStart) gazeAwayStreakStart = now

  const awaySec = (now - gazeAwayStreakStart) / 1000
  if (awaySec >= 5) return { category: 'GAZE', severity: 'HIGH', confidence: Math.min(1, awaySec / 5), explanation: `Looking away for ${awaySec.toFixed(1)}s` }
  if (awaySec >= 3) return { category: 'GAZE', severity: 'MEDIUM', confidence: Math.min(1, awaySec / 3), explanation: `Looking away for ${awaySec.toFixed(1)}s` }
  if (awaySec >= 1) return { category: 'GAZE', severity: 'LOW', confidence: Math.min(1, awaySec / 1), explanation: `Looking away for ${awaySec.toFixed(1)}s` }
  return null
}

function triggerIncident(category, incident) {
  // Use the category evaluateIncident is tracking, not incident.category —
  // phone/device incidents come straight from worker.js and if that
  // object's category field is ever missing or spelled differently, keying
  // off it here would silently stash the entry under the wrong (or
  // undefined) key, so alreadyActive/resolve checks elsewhere would never
  // find it again and the incident could never be tracked or closed.
  const prev = activeIncidents[category]
  if (prev) {
    if (prev.severity === incident.severity) return
    resolveIncident(category)
  }

  const otherActive = Object.keys(activeIncidents).filter(c => c !== category)
  const correlated = otherActive.length > 0
  const severity = correlated ? 'HIGH' : incident.severity
  const explanation = correlated
    ? `${incident.explanation} — overlapping with ${otherActive.join(', ')}`
    : incident.explanation

  const id = Date.now()
  activeIncidents[category] = { id, severity }

  const msg = {
    id,
    category,
    severity,
    confidence: incident.confidence,
    explanation,
    name: category.toLowerCase(),
    flagType: category,
    description: explanation,
    correlated,
    startTime: suspicionStartTime || window.realTime(),
    timestamp: performance.now() * 1000,
    belief: incident.confidence,
    proof: typeof window.captureProof === 'function' ? window.captureProof() : null
  }

  sendToProctor('incident_open', { incident: serializeIncidentForWire(msg) })
  if (msg.proof && typeof msg.proof.close === 'function') msg.proof.close()
}

function resolveIncident(category) {
  const active = activeIncidents[category]
  if (!active) return
  sendToProctor('incident_close', {
    id: active.id,
    startedAt: suspicionStartTime || window.realTime(),
    timestamp: window.realTime(),
    belief: currentRisk / 100
  })
  delete activeIncidents[category]
}

function triggerPositionFlag(context) {
  const flagId = Date.now()
  activePositionFlag = true
  lastPositionFlagId = flagId
  lastPositionFlagTime = flagId
  positionClearStartedAt = null

  const msg = {
    id: flagId,
    name: 'position_anomaly',
    flagType: context.type,
    emoji: context.emoji,
    description: context.description,
    correlated: context.correlated,
    timestamp: performance.now() * 1000,
    startTime: suspicionStartTime || window.realTime(),
    belief: currentRisk / 100,
    proof: null,
    severity: context.severity
  }

  sendToProctor('incident_open', { incident: serializeIncidentForWire(msg) })
}

function resolvePositionFlag() {
  const id = lastPositionFlagId
  if (!id) return
  sendToProctor('incident_close', {
    id,
    startedAt: suspicionStartTime || window.realTime(),
    timestamp: window.realTime(),
    belief: currentRisk / 100
  })
  activePositionFlag = false
  positionClearStartedAt = null
  lastPositionFlagId = null
}

function generateFlagContext() {
  const total = signals.framesAnalyzed || 1
  const absentRatio = signals.faceAbsentCount / total
  const phoneRatio = signals.phoneDetectedCount / total
  const deviceRatio = signals.deviceDetectedCount / total

  if (phoneRatio > 0.3 && absentRatio > 0.3) {
    return {
      type: "CRITICAL", emoji: "🔴", severity: "HIGH",
      description: `Phone detected${lastPhoneZone ? " at " + lastPhoneZone : ""} while candidate absent — correlated threat`,
      correlated: true
    }
  }
  if (signals.multipleFacesCount > 2 && phoneRatio > 0.1) {
    return {
      type: "CRITICAL", emoji: "🔴", severity: "HIGH",
      description: "Multiple persons and device detected simultaneously — correlated threat",
      correlated: true
    }
  }
  if (phoneRatio > 0.3) {
    return {
      type: "DEVICE", emoji: "🟠", severity: "HIGH",
      description: `Phone visible${lastPhoneZone ? " at " + lastPhoneZone + " of frame" : ""} for ${conditionDurations.phoneVisible} seconds`,
      correlated: false
    }
  }

if (signals.multipleFacesCount > 2 && deviceRatio > 0.3) {
    return {
      type: "CRITICAL", emoji: "🔴", severity: "HIGH",
      description: `Second person detected alongside ${lastDeviceClass || 'device'} — correlated threat`,
      correlated: true
    }
  }

  if (deviceRatio > 0.3) {
    return {
      type: "DEVICE", emoji: "🟠", severity: "HIGH",
      description: `${lastDeviceClass ? lastDeviceClass.charAt(0).toUpperCase() + lastDeviceClass.slice(1) : 'Electronic device'} detected${lastDeviceZone ? " at " + lastDeviceZone + " of frame" : ""} — unauthorized device visible`,
      correlated: false
    }
  }


  if (signals.multipleFacesCount > 2) {
    const zone = lastPersonZones[1] ? " at " + lastPersonZones[1] + " of frame" : ""
    return {
      type: "PRESENCE", emoji: "🟡", severity: "MEDIUM",
      description: `Second person detected${zone} for ${conditionDurations.multiPerson} consecutive seconds`,
      correlated: false
    }
  }
  if (absentRatio > 0.5) {
    return {
      type: "PRESENCE", emoji: "🟡", severity: "MEDIUM",
      description: `Candidate absent from frame for ${conditionDurations.personAbsent} seconds`,
      correlated: false
    }
  }
  const gazeRatio = signals.gazeAwayCount / (signals.framesAnalyzed || 1)
  if (gazeRatio > 0.3) {
    return {
      type: "BEHAVIORAL", emoji: "🔵", severity: "LOW",
      description: `Candidate gaze away ${Math.round(gazeRatio * 100)}% of the time — sustained distraction`,
      correlated: false
    }
  }

  return {
    type: "ANOMALY", emoji: "⚪", severity: "LOW",
    description: "Suspicious behavioral pattern detected",
    correlated: false
  }
}

export function checkForFlag() {
  const now = Date.now()

  if (currentRisk > 60) {
    consecutiveHighTicks++
    consecutiveLowTicks = 0
  } else if (currentRisk < 30) {
    consecutiveLowTicks++
    consecutiveHighTicks = 0
  } else {
    consecutiveHighTicks = 0
    consecutiveLowTicks = 0
  }

  const cooldownPassed = now - lastIncidentCloseTime > INCIDENT_COOLDOWN_MS

  const anyCategoryActive = Object.keys(activeIncidents).length > 0
  if (consecutiveHighTicks >= CONFIRMATION_TICKS && cooldownPassed && !activeIncidentId && !anyCategoryActive) {
    activeIncidentId = Date.now()
    const context = generateFlagContext()
    incidentStartTime = performance.now() * 1000

    const msg = {
      id: activeIncidentId,
      name: getIncidentName(),
      flagType: context.type,
      emoji: context.emoji,
      description: context.description,
      correlated: context.correlated,
      timestamp: incidentStartTime,
      startTime: suspicionStartTime || window.realTime(),
      belief: currentRisk / 100,
      proof: typeof window.captureProof === 'function' ? window.captureProof() : null,
      severity: context.severity
    }

    sendToProctor('incident_open', { incident: serializeIncidentForWire(msg) })
    if (msg.proof && typeof msg.proof.close === 'function') msg.proof.close()
  } else if (consecutiveLowTicks >= CONFIRMATION_TICKS && activeIncidentId && !activeBehavioralFlag) {
    const closeProof = typeof window.captureProof === 'function' ? window.captureProof() : null
    sendToProctor('incident_close', {
      id: activeIncidentId,
      startedAt: suspicionStartTime || window.realTime(),
      timestamp: lastSuspiciousTime || window.realTime(),
      belief: currentRisk / 100,
      proof: closeProof ? bitmapToDataURL(closeProof) : null
    })
    if (closeProof && typeof closeProof.close === 'function') closeProof.close()
    activeIncidentId = null
    lastIncidentCloseTime = now
  }
}

export function updateSignals(results, phoneRawConf = 0, phoneZone = null, personZones = [], rawPersonCount = 0, deviceRawConf = 0, deviceZone = null, deviceClass = null, phoneIncident = null, deviceIncident = null) {
  signals.framesAnalyzed++
  const total = signals.framesAnalyzed || 1

const extraPersonActive = rawPersonCount > 1 || results.some(r => r.condition === 'EXTRA_PERSON' && r.state !== 'OK')
const anyPersonDetected = rawPersonCount >= 1  
const phoneActive = phoneRawConf > 0.35

results.forEach(r => {
  if (r.condition === 'PERSON_MISSING' && r.state !== 'OK' && !extraPersonActive && !phoneActive && !anyPersonDetected) signals.faceAbsentCount++
  if (r.condition === 'EXTRA_PERSON' && r.state !== 'OK') signals.multipleFacesCount++
  if (r.condition === 'GAZE_AWAY' && r.state === 'AWAY') signals.gazeAwayCount++
})

// frame-level counters belong outside the per-tracker loop
if (phoneRawConf > 0.35) signals.phoneDetectedCount++
if (deviceRawConf > 0.35) signals.deviceDetectedCount++

  if (phoneZone) lastPhoneZone = phoneZone
  if (personZones.length > 0) lastPersonZones = personZones
  if (deviceZone) lastDeviceZone = deviceZone
  if (deviceClass) lastDeviceClass = deviceClass


  if (signals.phoneDetectedCount > 0) conditionDurations.phoneVisible += (samplingInterval / 1000)
  else conditionDurations.phoneVisible = 0

  if (signals.faceAbsentCount > 0) conditionDurations.personAbsent += (samplingInterval / 1000)
  else conditionDurations.personAbsent = 0

  if (signals.multipleFacesCount > 0) conditionDurations.multiPerson += (samplingInterval / 1000)
  else conditionDurations.multiPerson = 0
const multiIncident = rawPersonCount > 1
    ? { category: 'MULTI', severity: 'MEDIUM', confidence: Math.min(1, (rawPersonCount - 1) * 0.5),
        explanation: `${rawPersonCount} persons detected in frame` }
    : null
  const absentIncident = (results.some(r => r.condition === 'PERSON_MISSING' && r.state === 'RED') && !extraPersonActive && !phoneActive && !anyPersonDetected)
    ? { category: 'PRESENCE', severity: 'MEDIUM', confidence: 0.8,
        explanation: 'Candidate not visible in frame' }
    : null
  evaluateIncident('PHONE', phoneIncident)
  evaluateIncident('DEVICE', deviceIncident)
  evaluateIncident('MULTI', multiIncident)
  evaluateIncident('PRESENCE', absentIncident)

  const isSuspicious = signals.faceAbsentCount > 0 ||
    signals.phoneDetectedCount > 0 ||
    signals.multipleFacesCount > 0
  if (isSuspicious) lastSuspiciousTime = window.realTime()

  sendToProctor('belief_update', { id: 'bar-person', value: signals.faceAbsentCount / total })
  sendToProctor('belief_update', { id: 'bar-multi', value: signals.multipleFacesCount / total })
  sendToProctor('belief_update', { id: 'bar-phone', value: signals.phoneDetectedCount / total })
  sendToProctor('belief_update', { id: 'bar-device', value: signals.deviceDetectedCount / total })
  sendToProctor('pill_update', { id: 'pill-person', active: signals.faceAbsentCount / total > 0.3 })
  sendToProctor('pill_update', { id: 'pill-multi', active: signals.multipleFacesCount > 0 })
  sendToProctor('pill_update', { id: 'pill-phone', active: signals.phoneDetectedCount / total > 0.1 })
  //console.log("signals:", JSON.stringify(signals), "risk:", currentRisk)
}

function accumulateRisk(windowScore, dt) {
  const pull = RISK_ACCUMULATE_RATE * (dt / SCORE_INTERVAL_MS)
  currentRisk = Math.min(100, Math.max(windowScore, currentRisk + windowScore * pull))
}

function decayRisk(dt) {
  const decayPerMs = Math.pow(0.5, 1 / RISK_DECAY_HALF_LIFE_MS)
  currentRisk = currentRisk * Math.pow(decayPerMs, dt)
  if (currentRisk < 10) suspicionStartTime = null
}

export function updateRiskScore(windowScore) {
  const now = Date.now()
  const dt = now - lastRiskUpdateTime
  lastRiskUpdateTime = now

  if (windowScore > 0) {
    accumulateRisk(windowScore, dt)
  } else {
    decayRisk(dt)
  }

  currentRisk = Math.round(currentRisk)
  sendToProctor('risk_update', { risk: currentRisk })

  if (currentRisk >= 30 && !suspicionStartTime) {
    suspicionStartTime = window.realTime()
  }

  updateSamplingRate()
  checkForFlag()
}

export function getIncidentName() {
  const total = signals.framesAnalyzed || 1
  if (signals.faceAbsentCount / total > 0.3) return "no_person_visible"
  if (signals.phoneDetectedCount / total > 0.1) return "cell_phone_visible"
  if (signals.multipleFacesCount > 2) return "multiple_persons"
  return "no_person_visible"
}

export function triggerBehavioralFlag(context) {
  if (activeIncidentId) return
  activeIncidentId = Date.now()
  incidentStartTime = performance.now() * 1000
  activeBehavioralFlag = true
  lastBehavioralFlagId = activeIncidentId

  const msg = {
    id: activeIncidentId,
    name: "no_person_visible",
    flagType: context.type,
    emoji: context.emoji,
    description: context.description,
    correlated: context.correlated,
    timestamp: incidentStartTime,
    startTime: suspicionStartTime || window.realTime(),
    belief: currentRisk / 100,
    proof: null,
    severity: context.severity
  }

  sendToProctor('incident_open', { incident: serializeIncidentForWire(msg) })
}

// in risk-engine.js, near your other UI helpers
function updateGazeUI(gaze) {
  //console.log('gaze state:', gaze.state, 'score:', gaze.score)
  const belief = Math.max(0, Math.min(1, (typeof gaze.score === 'number' ? gaze.score : 0) / 100));
  sendToProctor('belief_update', { id: 'bar-gaze', value: belief })
  // Full gaze_update carries the richer state the proctor UI needs
  // (state label, event counts, face position) beyond just the belief bar.
  sendToProctor('gaze_update', {
    state: gaze.state,
    score: gaze.score,
    awayEventCount: gaze.awayEventCount,
    awayDuration60: gaze.awayDuration60,
    oscillations30: gaze.oscillations30,
    faceX: gaze.faceX,
    faceY: gaze.faceY
  })
}

export function updateGazeSignals(gaze) {
  updateGazeUI(gaze); // add this first

  if (gaze.state === 'away' || gaze.state === 'absent') {
    signals.gazeAwayCount++;
  } else if (gaze.state === 'unstable') {
    signals.gazeAwayCount += 0.5;
  }

  if (typeof gaze.score === 'number') {
    gazeScoreAccum += gaze.score;
    gazeScoreSamples++;
  }

  lastGazeMsg = gaze;
  updateFacePosition(gaze.faceX, gaze.faceY);
const gazeIncident = classifyGazeAway(gaze)
  evaluateIncident('GAZE', gazeIncident)
}

function checkForGazeFlag(gaze) {
  const now = Date.now()
  const cooldownPassed = !lastGazeFlagTime || (now - lastGazeFlagTime > GAZE_FLAG_COOLDOWN)

  // Determine if any gaze threshold is currently breached
  const awayDurationBreach  = gaze.awayDuration60 > 10
  const awayEventsBreach    = gaze.awayEventCount >= 3
  const oscillationBreach   = gaze.oscillations30 >= 6
  const anyBreach = awayDurationBreach || awayEventsBreach || oscillationBreach

  if (anyBreach && !activeGazeFlag && cooldownPassed) {
    // Build description for the most severe breach
    let description
    if (oscillationBreach && awayDurationBreach) {
      description = `Rapid gaze switching (${gaze.oscillations30} switches/30s) with ${gaze.awayDuration60}s cumulative away time — possible secondary screen`
    } else if (oscillationBreach) {
      description = `Rapid gaze switching detected — ${gaze.oscillations30} direction changes in 30 seconds (possible secondary screen)`
    } else if (awayEventsBreach && awayDurationBreach) {
      description = `Candidate looked away ${gaze.awayEventCount} times for ${gaze.awayDuration60}s total in the last minute — repeated pattern`
    } else if (awayDurationBreach) {
      description = `Candidate looked away for ${gaze.awayDuration60} seconds in the last minute — exceeds threshold`
    } else {
      description = `Candidate looked away ${gaze.awayEventCount} times in 60 seconds — repeated distraction pattern`
    }

    triggerGazeFlag({
      type: 'BEHAVIORAL',
      emoji: '🔵',
      severity: 'LOW',
      description,
      correlated: false
    })
  } else if (activeGazeFlag && !anyBreach) {
    // Start/extend resolve grace period
    if (!gazeClearStartedAt) gazeClearStartedAt = now
    if (now - gazeClearStartedAt >= GAZE_RESOLVE_GRACE) {
      resolveGazeFlag()
    }
  } else if (activeGazeFlag && anyBreach) {
    // Still breached — reset resolve grace
    gazeClearStartedAt = null
  }
}

function triggerGazeFlag(context) {
  // Gaze flags are independent of position-based behavioral flags — use separate ID tracking
  const flagId = Date.now()
  activeGazeFlag = true
  lastGazeFlagId = flagId
  lastGazeFlagTime = flagId
  gazeClearStartedAt = null

  const msg = {
    id: flagId,
    name: 'gaze_anomaly',
    flagType: context.type,
    emoji: context.emoji,
    description: context.description,
    correlated: context.correlated,
    timestamp: performance.now() * 1000,
    startTime: suspicionStartTime || window.realTime(),
    belief: currentRisk / 100,
    proof: null,
    severity: context.severity
  }
  console.log("🔥 INCIDENT OPEN", msg)

  sendToProctor('incident_open', { incident: serializeIncidentForWire(msg) })
}

function resolveGazeFlag() {
  const id = lastGazeFlagId
  if (!id) return
  sendToProctor('incident_close', {
    id,
    startedAt: suspicionStartTime || window.realTime(),
    timestamp: window.realTime(),
    belief: currentRisk / 100
  })
  activeGazeFlag = false
  gazeClearStartedAt = null
  lastGazeFlagId = null
}

function resolveBehavioralFlag() {
  const id = lastBehavioralFlagId
  if (!id) return
  sendToProctor('incident_close', {
    id,
    startedAt: suspicionStartTime || window.realTime(),
    timestamp: window.realTime(),
    belief: currentRisk / 100
  })
  activeBehavioralFlag = false
  behavioralClearStartedAt = null
  activeIncidentId = null
  lastBehavioralFlagId = null
}

let lastWindowSnapshot = null  // last completed window — what the UI should show

setInterval(() => {
  const windowScore = scoreWindow()
  // Snapshot before reset so the UI always has a complete window to display
  lastWindowSnapshot = { ...signals }
  updateRiskScore(windowScore)
  signals = {
    framesAnalyzed: 0,
    faceAbsentCount: 0,
    multipleFacesCount: 0,
    gazeAwayCount: 0,
    phoneDetectedCount: 0,
    deviceDetectedCount: 0
  }
  gazeScoreAccum = 0
  gazeScoreSamples = 0
}, SCORE_INTERVAL_MS)

window._signals      = () => signals
window._lastSnapshot = () => lastWindowSnapshot || signals  // falls back to live if no window completed yet
window._currentRisk = () => currentRisk
window._getRisk = () => currentRisk
window._setSignals = (key, val) => { signals[key] = val }
window._conditionDurations = () => conditionDurations
window._homePosition = () => homePosition
window._positionFlag = () => ({ active: activePositionFlag, id: lastPositionFlagId })
window._lastGazeMsg = () => lastGazeMsg
window._gazeFlag = () => ({ active: activeGazeFlag, id: lastGazeFlagId })