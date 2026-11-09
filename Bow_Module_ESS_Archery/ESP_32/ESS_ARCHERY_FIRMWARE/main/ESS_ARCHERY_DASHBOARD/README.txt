ESS-ARCHERY Field App V1.4 Coordinate Target

Changes from V1.3:
- Replaces manual 1-10 score buttons with an interactive target coordinate selector.
- Tap arrow landing position after STOP/COMPLETE.
- Coordinates: target center (0,0), +X right, +Y up.
- Stores X/Y in normalized target-radius units and centimeters.
- Computes radial error and an automatic ring score from the selected coordinate.
- Drona analytics now use radial error as the outcome instead of manually entered score.
- Adds horizontal/vertical landing-bias reporting.
- Haptic-active plus 200 ms recovery-guard samples remain in raw data but are excluded from natural-stability summary metrics.
- Summary CSV now includes coordinate, radial-error, sample-count and effective sample-rate fields.
- Raw CSV repeats coordinate metadata for every sample.
- Loads legacy V1.3 local data when V1.4 storage is empty; old score-only shots are not used for coordinate analytics.

Deployment:
Copy index.html, manifest.webmanifest, service-worker.js and your existing archer_logo.png into the Cloudflare deployment folder.
Use HTTPS. On Android use Chrome/Chromium for Web Bluetooth.
