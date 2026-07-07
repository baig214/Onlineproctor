'use strict';

// ─── Constants (mirrors app.py) ───────────────────────────────────────────────

const WARMUP_FRAMES            = 3;
const MIN_BLOB_AREA            = 1500;
const MIN_MOTION_RATIO         = 0.01;
const ZONE_SUPPRESSION_SECONDS = 30;

// ─── Module-level state ───────────────────────────────────────────────────────

let frameCount        = 0;

/** @type {Uint8ClampedArray|null} RGBA pixels of previous frame */
let previousFrameData = null;
let previousWidth     = 0;
let previousHeight    = 0;

/** @type {Map<string, number>} zone name → last-alert timestamp (s) */
let recentZoneAlerts  = new Map();

// ─── Reusable typed-array buffers (allocated once, reused every frame) ────────

let grayPrev     = null;   // Uint8Array [w*h]
let grayCurr     = null;
let diffMask     = null;
let candidateMask  = null;
let filteredMask   = null;
let validMask      = null;
let morphScratch   = null; // scratch for morphology

// ─── Entry point ──────────────────────────────────────────────────────────────

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === 'reset') {
    resetState();
    return;
  }

  if (msg.type === 'frame') {
    try {
      const result = await processFrame(msg.frame, msg.personBox ?? null);
      if (result !== null && result.motionDetected) {
        // Transfer bitmaps to avoid copying — they're neutered after postMessage
        self.postMessage({ type: 'motion', ...result }, [result.beforeBitmap, result.afterBitmap, result.diffBitmap]);      }
    } catch (err) {
      console.error('[motion-worker]', err);
    } finally {
      // VideoFrame must always be closed to release the underlying resource
      msg.frame.close();
    }
  }
};

// ─── State reset ──────────────────────────────────────────────────────────────

function resetState() {
  frameCount        = 0;
  previousFrameData = null;
  previousWidth     = 0;
  previousHeight    = 0;
  recentZoneAlerts  = new Map();
  // Null buffers so ensureBuffers() re-allocates if dimensions change
  grayPrev = grayCurr = diffMask = null;
  candidateMask = filteredMask = validMask = morphScratch = null;
}

// ─── Main pipeline ────────────────────────────────────────────────────────────

/**
 * Returns the motion result object, or null during warmup.
 *
 * @param {VideoFrame} videoFrame
 * @param {{x1:number,y1:number,x2:number,y2:number}|null} personBox
 */
async function processFrame(videoFrame, personBox) {
  // ── Phase 1: decode VideoFrame → RGBA pixels via OffscreenCanvas ────────────
  const { rgba, width, height } = await decodeVideoFrame(videoFrame);

  // ── Phase 2: state ──────────────────────────────────────────────────────────
  const priorData   = previousFrameData;
  const priorWidth  = previousWidth;
  const priorHeight = previousHeight;

  frameCount++;
  const currentCount = frameCount;

  previousFrameData = rgba.slice();
  previousWidth     = width;
  previousHeight    = height;

  if (currentCount <= WARMUP_FRAMES || priorData === null) {
    return null; // warmup — nothing to report
  }

  // ── Phase 3: candidate mask ─────────────────────────────────────────────────
  ensureBuffers(width, height);
  buildCandidateMask(personBox, width, height, candidateMask);
  const outsideCandidatePixels = countNonZero(candidateMask, width * height);
keepLargeComponents
  // ── Phase 4: frame differencing ─────────────────────────────────────────────
  toGray(priorData, grayPrev, width * height);
  toGray(rgba,      grayCurr, width * height);
  computeFrameDiff(grayPrev, grayCurr, candidateMask, diffMask, width * height);
  const diffSum = diffMask.reduce((s,v) => s+v, 0)
//console.log('diffSum after frameDiff:', diffSum, 'changedPx after components:', 0)

  // ── Phase 5: morphological filter ───────────────────────────────────────────
  filterMotionMask(diffMask, candidateMask, filteredMask, morphScratch, width, height);

  // ── Phase 6: connected components ───────────────────────────────────────────
  const { changedPixels } = keepLargeComponents(filteredMask, validMask, width, height);
  //console.log('changedPx after components:', changedPixels,)

  // ── Phase 7: ratio + zones + suppression ────────────────────────────────────
  const motionRatio   = computeMotionRatio(changedPixels, outsideCandidatePixels);
  const activeZones   = getActiveZones(validMask, width, height);
  //console.log('changedPixels:', changedPixels, 'motionRatio:', motionRatio, 'zones:', activeZones);
  const motionPresent = changedPixels > MIN_BLOB_AREA && motionRatio > MIN_MOTION_RATIO;
  const motionDetected = motionPresent && shouldAlertForZones(activeZones);

  // ── Phase 8: render diff image ───────────────────────────────────────────────
  const diffImageData = renderDiff(rgba, validMask, width, height);

  // ── Phase 9: produce ImageBitmaps (no JPEG — drawn directly to canvas) ───────
  const [beforeBitmap, afterBitmap, diffBitmap] = await Promise.all([
    createImageBitmap(new ImageData(new Uint8ClampedArray(priorData), priorWidth, priorHeight)),
    createImageBitmap(new ImageData(rgba, width, height)),
    createImageBitmap(diffImageData),
  ]);

  return {
    motionDetected,
    motionRatio: Math.round(motionRatio * 1e6) / 1e6,
    changedPx: changedPixels,
    // activeZones as [{name}] to match showMotionToast expectation
    activeZones: activeZones.map(name => ({ name })),
    isFlash: false,          // flash detection is a separate signal — not in app.py
    flashZones: [],
    beforeBitmap,
    afterBitmap,
    diffBitmap,
  };
}

// ─── Utilities ────────────────────────────────────────────────────────────────

function ensureBuffers(width, height) {
  const n = width * height;
  if (!grayPrev     || grayPrev.length     !== n) grayPrev      = new Uint8Array(n);
  if (!grayCurr     || grayCurr.length     !== n) grayCurr      = new Uint8Array(n);
  if (!diffMask     || diffMask.length     !== n) diffMask      = new Uint8Array(n);
  if (!candidateMask || candidateMask.length !== n) candidateMask = new Uint8Array(n);
  if (!filteredMask  || filteredMask.length  !== n) filteredMask  = new Uint8Array(n);
  if (!validMask     || validMask.length     !== n) validMask     = new Uint8Array(n);
  if (!morphScratch  || morphScratch.length  !== n) morphScratch  = new Uint8Array(n);
}

function countNonZero(arr, n) {
  let c = 0;
  for (let i = 0; i < n; i++) if (arr[i] !== 0) c++;
  return c;
}

// ─── Phase stubs (replaced one phase at a time) ───────────────────────────────

// ─── Phase 1: decode VideoFrame → RGBA ───────────────────────────────────────
//
// Python equivalent: decode_jpeg_frame + validate_dimensions
//
// Key differences from Python:
//   - Input is already a VideoFrame (no base64 decoding needed)
//   - Output is RGBA, not BGR — all downstream math uses RGBA throughout
//   - Gray conversion formula must match OpenCV COLOR_BGR2GRAY weights:
//       OpenCV: gray = 0.114·B + 0.587·G + 0.299·R
//       Here (RGBA input): gray = 0.299·R + 0.587·G + 0.114·B   (same weights)
//   - OffscreenCanvas is reused across calls when dimensions are stable

/** @type {OffscreenCanvas|null} */
let _decodeCanvas = null;
/** @type {OffscreenCanvasRenderingContext2D|null} */
let _decodeCtx    = null;

/**
 * Draws a VideoFrame into an OffscreenCanvas and reads back RGBA pixels.
 * Mirrors decode_jpeg_frame + validate_dimensions from app.py.
 *
 * @param {VideoFrame} vf
 * @returns {{ rgba: Uint8ClampedArray, width: number, height: number }}
 */
async function decodeVideoFrame(vf) {
  const width  = vf.displayWidth;
  const height = vf.displayHeight;

  if (width <= 0 || height <= 0) {
    throw new Error(`Invalid VideoFrame dimensions: ${width}x${height}`);
  }

  // Reuse canvas when dimensions are unchanged (common case)
  if (!_decodeCanvas || _decodeCanvas.width !== width || _decodeCanvas.height !== height) {
    _decodeCanvas = new OffscreenCanvas(width, height);
    _decodeCtx    = _decodeCanvas.getContext('2d', {
      willReadFrequently: true,   // hint to browsers to keep pixels in CPU memory
      alpha: false,               // we never need transparency; saves a composite step
    });
  }

  _decodeCtx.drawImage(vf, 0, 0);

  // getImageData returns a Uint8ClampedArray in RGBA order, 4 bytes per pixel
  const imageData = _decodeCtx.getImageData(0, 0, width, height);
  return { rgba: imageData.data, width, height };
}

function buildCandidateMask(personBox, width, height, out) {
  out.fill(255);
  if (personBox === null) return;

  const { x1, y1, x2, y2 } = personBox;
  if (!isFinite(x1) || !isFinite(y1) || !isFinite(x2) || !isFinite(y2)) return;

  let left  = Math.floor(Math.min(x1, x2));
  let right = Math.ceil(Math.max(x1, x2));

const bboxWidthRatio = (right - left) / width
const padFactor = bboxWidthRatio > 0.6 ? 0 : bboxWidthRatio > 0.35 ? 0.10 : 0.15
const padX = Math.floor((right - left) * padFactor)
  left  = Math.max(left  - padX, 0);
  right = Math.min(right + padX, width);

  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = left; x < right; x++) {
      out[row + x] = 0;
    }
  }

  
}

function toGray(rgba, out, n) {
  for (let i = 0; i < n; i++) {
    const j = i * 4;
    out[i] = (rgba[j] * 0.299 + rgba[j+1] * 0.587 + rgba[j+2] * 0.114) | 0;
  }
}

function computeFrameDiff(prev, curr, mask, out, n) {
  for (let i = 0; i < n; i++) {
    out[i] = (mask[i] === 255 && Math.abs(curr[i] - prev[i]) > 30) ? 255 : 0;
  }
}

function filterMotionMask(diff, mask, out, scratch, width, height) {
  erode(diff, scratch, width, height, 5);
  dilate(scratch, out, width, height, 5);  // open = erode then dilate

  dilate(out, scratch, width, height, 5);
  erode(scratch, out, width, height, 5);  // close = dilate then erode

  // AND with candidate mask (mirrors final bitwise_and in Python)
  const n = width * height;
  for (let i = 0; i < n; i++) {
    if (mask[i] === 0) out[i] = 0;
  }
}

function erode(src, out, width, height, ksize) {
  const half = (ksize - 1) >> 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let min = 255;
      for (let ky = -half; ky <= half; ky++) {
        const ny = y + ky;
        if (ny < 0 || ny >= height) { min = 0; break; }
        for (let kx = -half; kx <= half; kx++) {
          const nx = x + kx;
          if (nx < 0 || nx >= width) { min = 0; break; }
          if (src[ny * width + nx] === 0) { min = 0; break; }
        }
        if (min === 0) break;
      }
      out[y * width + x] = min;
    }
  }
}

function dilate(src, out, width, height, ksize) {
  const half = (ksize - 1) >> 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let max = 0;
      outer: for (let ky = -half; ky <= half; ky++) {
        const ny = y + ky;
        if (ny < 0 || ny >= height) continue;
        for (let kx = -half; kx <= half; kx++) {
          const nx = x + kx;
          if (nx < 0 || nx >= width) continue;
          if (src[ny * width + nx] === 255) { max = 255; break outer; }
        }
      }
      out[y * width + x] = max;
    }
  }
}

function keepLargeComponents(mask, out, width, height) {
  out.fill(0);
  const n = width * height;
  const labels = new Int32Array(n);
  let nextLabel = 1;
  const areas = [];

  // 8-connected BFS labelling
  const queue = new Int32Array(n);
  const dx = [-1, 0, 1, -1, 1, -1, 0, 1];
  const dy = [-1, -1, -1, 0, 0, 1, 1, 1];

  for (let i = 0; i < n; i++) {
    if (mask[i] !== 255 || labels[i] !== 0) continue;

    const label = nextLabel++;
    areas.push(0);
    let head = 0, tail = 0;
    queue[tail++] = i;
    labels[i] = label;

    while (head < tail) {
      const idx = queue[head++];
      areas[label - 1]++;
      const cy = (idx / width) | 0;
      const cx = idx % width;
      for (let d = 0; d < 8; d++) {
        const nx = cx + dx[d];
        const ny = cy + dy[d];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const ni = ny * width + nx;
        if (mask[ni] !== 255 || labels[ni] !== 0) continue;
        labels[ni] = label;
        queue[tail++] = ni;
      }
    }
  }

  // Keep only blobs >= MIN_BLOB_AREA
  let changedPixels = 0;
  for (let i = 0; i < n; i++) {
    const lbl = labels[i];
    if (lbl > 0 && areas[lbl - 1] >= MIN_BLOB_AREA) {
      out[i] = 255;
      changedPixels++;
    }
  }

  return { changedPixels };
}

function computeMotionRatio(changedPixels, outsideCandidatePixels) {
  if (outsideCandidatePixels <= 0) return 0;
  return changedPixels / outsideCandidatePixels;
}

function getActiveZones(mask, width, height) {
  const rowEdges = [0, height >> 1, height];
  const colEdges = [0, (width / 3) | 0, ((2 * width) / 3) | 0, width];
  const rowNames = ['top', 'bottom'];
  const colNames = ['left', 'center', 'right'];

  const active = [];
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 3; c++) {
      let found = false;
      outer: for (let y = rowEdges[r]; y < rowEdges[r + 1]; y++) {
        for (let x = colEdges[c]; x < colEdges[c + 1]; x++) {
          if (mask[y * width + x] === 255) { found = true; break outer; }
        }
      }
      if (found) active.push(`${rowNames[r]}-${colNames[c]}`);
    }
  }
  return active;
}

function shouldAlertForZones(activeZones) {
  const now = Date.now() / 1000;

  // Prune expired entries
  for (const [zone, ts] of recentZoneAlerts) {
    if (now - ts >= ZONE_SUPPRESSION_SECONDS) recentZoneAlerts.delete(zone);
  }

  for (const zone of activeZones) {
    if (!recentZoneAlerts.has(zone)) {
      recentZoneAlerts.set(zone, now);
      return true;
    }
  }
  return false;
}

function renderDiff(rgba, mask, width, height) {
  const out = new Uint8ClampedArray(rgba);
  const n = width * height;

  for (let i = 0; i < n; i++) {
    if (mask[i] !== 255) continue;
    const j = i * 4;
    out[j]     = (rgba[j]     * 0.55 + 255 * 0.45) | 0;  // R: blended with red
    out[j + 1] = (rgba[j + 1] * 0.55) | 0;                // G: dimmed
    out[j + 2] = (rgba[j + 2] * 0.55) | 0;                // B: dimmed
    // alpha unchanged
  }

  return new ImageData(out, width, height);
}