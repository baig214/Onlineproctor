# ProctorSense Monitoring System

## Overview

ProctorSense is a browser-based AI-powered online examination monitoring system that analyzes candidate behavior in real time using computer vision and behavioral analytics.

The system continuously monitors candidate activity through multiple detection pipelines including:

- Face presence detection
- Eye gaze tracking
- Phone detection
- Device detection
- Motion analysis
- Behavioral risk scoring

All signals are processed and aggregated into a centralized risk engine that generates risk scores, incidents, and visual monitoring insights for proctors.

---

# System Architecture

```text
┌─────────────────────────────┐
│      Browser Dashboard      │
│       index.html            │
└──────────────┬──────────────┘
               │
               ▼

 ┌──────────────────────────┐
 │      worker.js           │
 │ Object Detection         │
 └──────────┬───────────────┘
            │
            ▼

 ┌──────────────────────────┐
 │   risk-engine.js         │
 │ Risk Aggregation Layer   │
 └──────────┬───────────────┘
            │
 ┌──────────┴───────────┐
 │                      │
 ▼                      ▼

gaze-worker.js    motion-worker.js
 Eye Tracking      Motion Analysis

            │
            ▼

 ┌──────────────────────────┐
 │ Incident Generation      │
 │ Risk Score Updates       │
 │ Dashboard Analytics      │
 └──────────────────────────┘
```

---

# Project Components

| File | Responsibility |
|--------|---------------|
| `index.html` | Dashboard interface and system orchestration |
| `worker.js` | Object detection and candidate monitoring |
| `gaze-worker.js` | Eye gaze and attention analysis |
| `motion-worker.js` | Motion detection and movement analysis |
| `risk-engine.js` | Risk scoring and incident management |

---

# Module Documentation

## index.html

### Purpose

Acts as the primary monitoring dashboard and user interface.

### Responsibilities

- Displays webcam feed
- Visualizes risk score
- Displays detection status
- Shows active incidents
- Maintains incident timeline
- Communicates with detection workers

### Major Components

#### Camera Panel

Displays:

- Candidate webcam feed
- Recording status
- FPS information
- Monitoring indicators

#### Risk Dashboard

Displays:

- Current risk score
- Risk trend visualization
- Session monitoring status
- Risk cycle information

#### Detection Grid

Provides real-time monitoring information for:

- Face presence
- Multiple faces
- Phone detection
- Device detection
- Eye gaze
- Motion analysis

#### Incident Timeline

Stores and visualizes:

- Open incidents
- Resolved incidents
- Event history
- Evidence snapshots

---

## worker.js

### Purpose

Performs real-time object detection using computer vision models.

### Responsibilities

#### Person Detection

Identifies:

- Candidate presence
- Candidate absence
- Multiple candidates

#### Phone Detection

Identifies:

- Mobile phones
- Phone activity
- Screen visibility

#### Device Detection

Identifies:

- Secondary screens
- Laptops
- Tablets
- External devices

### Processing Pipeline

```text
Video Frame
     ↓
Preprocessing
     ↓
Model Inference
     ↓
Object Classification
     ↓
Detection Event
     ↓
Risk Engine
```

---

## gaze-worker.js

### Purpose

Tracks eye movement and attention direction using facial landmarks.

### Responsibilities

#### Eye Direction Detection

Classifies gaze direction into:

- Center
- Left
- Right
- Up
- Down

#### Attention Analysis

Measures:

- Continuous away duration
- Total away time
- Frequency of gaze shifts

#### Calibration

Creates a personalized baseline for each candidate to improve gaze accuracy.

### Statistical Analysis

Uses:

- Median calculations
- Median Absolute Deviation (MAD)
- Deviation scoring

to produce stable attention measurements.

---

## motion-worker.js

### Purpose

Analyzes motion within the camera frame.

### Responsibilities

Detects:

- Body movement
- Desk movement
- Environmental movement

### Processing Pipeline

```text
Frame A
Frame B
      ↓
Frame Difference
      ↓
Motion Mask
      ↓
Noise Reduction
      ↓
Region Analysis
      ↓
Motion Event
```

### Techniques Used

- Frame differencing
- Morphological filtering
- Motion ratio computation
- Active region detection

---

## risk-engine.js

### Purpose

Acts as the central behavioral intelligence and decision-making module.

### Responsibilities

#### Signal Aggregation

Combines signals from:

- Phone detection
- Device detection
- Face presence
- Eye tracking
- Motion analysis

#### Risk Scoring

Generates:

```text
Risk Score (0–100)
```

based on candidate behavior.

#### Incident Management

Creates and manages:

- Behavioral incidents
- Device incidents
- Presence incidents
- Attention incidents

#### Behavioral Analysis

Tracks:

- Attention drift
- Position drift
- Repeated violations
- Suspicious behavioral patterns

### Workflow Example

```text
Phone Appears
      ↓
worker.js
      ↓
Detection Event
      ↓
risk-engine.js
      ↓
Risk Score Update
      ↓
Incident Creation
      ↓
Dashboard Update
```

---

# Dependencies

```text
ONNX Runtime Web
MediaPipe Face Landmarker
Tabler Icons
Canvas API
Web Workers
Modern Browser APIs
```

---

# System Workflow

```text
Start Session
      ↓
Open Webcam
      ↓
Initialize Workers
      ↓
Process Video Frames
      ↓
Generate Detection Signals
      ↓
Risk Evaluation
      ↓
Incident Creation
      ↓
Dashboard Visualization
```

---

# Conclusion

ProctorSense is a real-time AI-assisted online examination monitoring platform that combines object detection, eye gaze analysis, motion tracking, and behavioral scoring into a unified monitoring system. The architecture uses dedicated worker modules for specialized analysis while a centralized risk engine aggregates signals and generates incidents, providing a comprehensive monitoring dashboard for exam proctors.