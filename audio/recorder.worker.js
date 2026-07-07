/**
 * recorder.worker.js  (classic worker)
 *
 * Receives messages from the main thread about recording cycles and speech
 * events, then decides whether to save or discard each clip.
 *
 * Save logic:
 *   wasExtended=false, speechInBase=false  → discard (silent 10s)
 *   wasExtended=false, speechInBase=true   → save base 10s buffer
 *   wasExtended=true,  speechInExt=true    → save full 20s buffer
 *   wasExtended=true,  speechInExt=false   → save base 10s portion (baseBuffer)
 *
 * FROM main:  { type: 'init',             mimeType }
 *             { type: 'window-start',     isExtension: bool }
 *             { type: 'cycle-end',        buffer, wasExtended, baseBuffer? }
 *             { type: 'speech-detected',  text, timestamp }
 *
 * TO main:    { type: 'status',  message }
 *             { type: 'save',    buffer, filename, mimeType }
 *             { type: 'extend' }
 */

let mimeType = 'video/webm';

let speechInBase    = false;
let speechInExt     = false;
let inExtension     = false;
let savedLabels     = new Set();

const GRACE_MS        = 3000;
let   windowStartedAt = null;

function makeLabel() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function save(buffer, label) {
  if (!buffer || !label)      return;
  if (savedLabels.has(label)) return;
  savedLabels.add(label);
  const copy     = buffer.slice(0);
  const filename = `clip_${label}.webm`;
  self.postMessage({ type: 'save', buffer: copy, filename, mimeType }, [copy]);
  self.postMessage({ type: 'status', message: `saved ${filename}` });
}

function resetCycle() {
  speechInBase = false;
  speechInExt  = false;
  inExtension  = false;
}

self.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === 'init') {
    mimeType = msg.mimeType;
    self.postMessage({ type: 'status', message: 'ready' });
  }

  if (msg.type === 'window-start') {
    windowStartedAt = Date.now();
    if (!msg.isExtension) {
      resetCycle();
      self.postMessage({ type: 'status', message: 'cycle started' });
    } else {
      inExtension = true;
      self.postMessage({ type: 'status', message: 'extension started' });
    }
  }

  if (msg.type === 'speech-detected') {
    self.postMessage({ type: 'status', message: `speech — "${msg.text}"${msg.isViolation ? ' [VIOLATION]' : ''}` });

    if (!inExtension) {
      if (!speechInBase) {
        speechInBase = true;
        self.postMessage({ type: 'extend' });
        self.postMessage({ type: 'status', message: 'speech in base → requesting extension' });
      }
    } else {
      const age = Date.now() - (windowStartedAt || 0);
      if (age <= GRACE_MS && !speechInBase) {
        speechInBase = true;
        self.postMessage({ type: 'status', message: `late speech (${age}ms) → attributed to base` });
      } else {
        speechInExt = true;
        self.postMessage({ type: 'status', message: 'speech in extension' });
      }
    }
  }

  if (msg.type === 'cycle-end') {
    const { buffer, wasExtended, baseBuffer } = msg;
    const label = makeLabel();

    if (!wasExtended) {
      if (speechInBase) {
        save(buffer, label);
        self.postMessage({ type: 'status', message: 'saved 10s (speech in base)' });
      } else {
        self.postMessage({ type: 'status', message: 'silent cycle — discarded' });
      }
    } else {
      if (speechInExt) {
        save(buffer, label);
        self.postMessage({ type: 'status', message: 'saved 20s clip (speech in extension)' });
      } else {
        if (baseBuffer) {
          save(baseBuffer, label);
          self.postMessage({ type: 'status', message: 'saved 10s only (no speech in extension)' });
        } else {
          save(buffer, label);
          self.postMessage({ type: 'status', message: 'saved full clip (no base snapshot available)' });
        }
      }
    }

    resetCycle();
  }
};