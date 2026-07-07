# 1. Persistent Candidate Tracking

## Overview

The monitoring system establishes a **primary candidate identity** at the beginning of the examination session and continuously maintains that identity throughout the session.

Instead of selecting the largest detected person in every frame, the system stores a persistent reference to the original candidate and attempts to match future detections against this reference.

This prevents identity switching when additional individuals enter the camera frame.

---

## Detection Workflow

```text
Session Start
      ↓
Person Detection
      ↓
Primary Candidate Selected
      ↓
Reference Bounding Box Stored
      ↓
Continuous Matching Across Frames
      ↓
Candidate Identity Maintained
```

---

## Stored Candidate State

The system maintains candidate-specific tracking information including:

- Initial bounding box coordinates
- Bounding box dimensions
- Candidate center position
- Historical movement data
- Detection confidence

Example state:

```javascript
candidateReference = {
    x,
    y,
    width,
    height,
    centerX,
    centerY
}
```

---

## Matching Logic

For every incoming frame, detected person bounding boxes are compared against the stored reference candidate.

Matching criteria include:

- Bounding box overlap
- Position proximity
- Relative size consistency
- Movement continuity

The candidate with the highest similarity score is retained as the primary subject.

---

## Purpose

This mechanism ensures:

- Stable candidate monitoring
- Reduced identity switching
- Accurate multi-person analysis
- Consistent behavioral tracking

---

# 2. Multiple Person Detection

## Overview

The system continuously evaluates the number of detected persons within each processed frame.

After identifying the primary candidate, all additional person detections are treated as secondary subjects.

The presence of secondary subjects may indicate unauthorized assistance or environmental interference.

---

## Detection Workflow

```text
Frame Received
      ↓
Person Detection
      ↓
Primary Candidate Identified
      ↓
Count Remaining Persons
      ↓
Generate Multi-Person Signal
```

---

## Detection Criteria

A multiple-person event is triggered when:

```text
Total Persons > 1
```

and at least one additional person remains visible beyond the configured persistence threshold.

---

## Persistence Filtering

Object detection models occasionally produce transient false positives.

To reduce false alarms, additional persons must remain visible across multiple consecutive detection cycles before an incident is generated.

Example logic:

```javascript
if (extraPersonFrames >= PERSON_PERSIST_THRESHOLD) {
    triggerMultiPersonEvent();
}
```

---

## Generated Signals

When triggered, the system produces:

- Multi-person detection signal
- Risk score contribution
- Incident timeline entry
- Evidence snapshot

---

# 3. Bounding Box Stabilization

## Overview

Raw object detection outputs naturally exhibit bounding box jitter caused by inference variation between frames.

Even when a subject remains stationary, bounding box coordinates may fluctuate slightly.

The stabilization layer smooths these fluctuations before the data is used by downstream systems.

---

## Problem

Raw detections:

```text
Frame 1 → x = 420
Frame 2 → x = 425
Frame 3 → x = 418
Frame 4 → x = 423
```

These variations create:

- Visual instability
- False motion readings
- Inaccurate position analysis

---

## Stabilization Process

The system combines current detections with previous tracking information.

```text
Previous Bounding Box
           +
Current Detection
           ↓
Smoothed Bounding Box
```

---

## Parameters Considered

The stabilization process evaluates:

- X position
- Y position
- Width
- Height
- Center coordinates

---

## Benefits

Bounding box stabilization improves:

- Candidate tracking accuracy
- Motion analysis reliability
- Head position calculations
- UI visualization quality

---

# 4. Phone Detection Filtering

## Overview

The object detection model is capable of identifying phones but may occasionally classify unrelated desk objects as mobile devices.

Common false positives include:

- Computer mice
- Remote controls
- USB accessories
- Small desk objects

The phone filtering system validates detections before they contribute to risk scoring.

---

## Bounding Box Validation

Each detected phone undergoes geometric analysis.

The system evaluates:

```text
Bounding Box Width
Bounding Box Height
Aspect Ratio
Detection Confidence
```

---

## Aspect Ratio Analysis

A phone typically exhibits a predictable shape.

The system calculates:

```javascript
aspectRatio = width / height;
```

The detected object must fall within an acceptable phone ratio range.

Objects outside the expected ratio window are discarded.

---

## Confidence Validation

The detection must also exceed a minimum confidence threshold before being accepted.

Example:

```javascript
if (
    confidence > PHONE_CONFIDENCE_THRESHOLD &&
    aspectRatioWithinRange
) {
    acceptDetection();
}
```

---

## Purpose

This filtering stage significantly reduces:

- Mouse misclassification
- Desk object misclassification
- False risk accumulation

---

# 5. Edge-Based Phone Detection

## Overview

Candidates may intentionally position a phone near the edge of the camera frame to reduce visibility.

Partial phone visibility often causes reduced model confidence and inconsistent detections.

The system compensates for this behavior by evaluating phone interaction with frame boundaries.

---

## Detection Logic

For every phone detection, the system checks whether the bounding box intersects the image border.

Example:

```text
Phone Bounding Box
         ↓
Touches Frame Edge?
         ↓
Apply Additional Suspicion Weight
```

---

## Evaluated Boundaries

The following conditions are checked:

```javascript
touchesLeftEdge
touchesRightEdge
touchesTopEdge
touchesBottomEdge
```

---

## Edge Contact Conditions

A phone is considered edge-adjacent when:

```javascript
bbox.x <= EDGE_THRESHOLD
```

or

```javascript
bbox.x + bbox.width >= frameWidth - EDGE_THRESHOLD
```

Similar checks are applied to vertical boundaries.

---

## Risk Contribution

Edge-adjacent phones may contribute additional risk weight compared to centrally visible devices because they are more likely to represent deliberate concealment attempts.

---

## Generated Outputs

When an edge-phone condition is detected, the system may:

- Generate a device-related flag
- Increase risk score contribution
- Capture evidence snapshots
- Record timeline events