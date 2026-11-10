# ESS-Archery — Electronic Stability System for Archery 🎯

**ESS-Archery** is an experimental sports-electronics project exploring how embedded sensing, motion analysis and computer vision can be combined to better understand an archer’s bow stability and shot outcome.

The system uses a custom **ESP32-based bow module**, BLE communication and a mobile/web dashboard to monitor bow motion during the aiming and holding phase. Voice-controlled **RECORD / STOP** commands allow trials to be captured without interrupting the shooting process.

The project has also been extended with a computer-vision pipeline using **OpenCV and ArUco-based perspective registration**. After a shot, a target image can be processed to estimate the arrow shaft and impact point and automatically calculate:

- Horizontal and vertical impact coordinates
- Radial error from target centre
- Estimated scoring ring
- Shot-placement data for later session analysis

Automatic results remain editable, with a manual target-entry method available when vision confidence is insufficient.

## Project Goal

The longer-term goal is not only to determine **where an arrow landed**, but to connect that result with **what happened to the bow before release**.

By combining stability metrics and shot-placement data across larger shooting sessions, ESS-Archery aims to explore patterns that may help archers better understand repeatability, movement and performance.

## Current Development Areas

The prototype currently explores:

- Embedded IMU-based bow stability measurement
- BLE communication and mobile field operation
- Voice-controlled shot recording
- Computer-vision arrow impact detection
- Shot coordinate and scoring analysis
- Session-level stability/impact analytics
- Future markerless vision and ML-assisted analysis

This is an ongoing experimental project, and collaboration or technical feedback from people working in **embedded systems, computer vision, machine learning, sports technology or archery** is very welcome.
