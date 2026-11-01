# Surveillance-motion experiments — protocol (frozen before any run)
Recorded: 2026-10-31T21:26Z

Base footage: static MOT17-02, -04, -09 (1920x1080, 30 fps). Virtual camera: pure rotation about the
optical centre, H = K R K^-1 (exact for any scene depth), focal length f = 1662.8 px (60 deg HFOV),
centred 1280x720 output window. GT boxes: corners mapped by H, axis-aligned bounding box, visibility
scaled by in-window fraction, <25% inside -> distractor, fully outside -> dropped. YOLOv8n detections
and features re-run once per variant and shared by all conditions.

1 PTZ sweep: pan between -8 and +8 deg at 5, 10, 20, 40 deg/s, 1 s dwell at each end, start at 0 moving right.
2 Pole sway: pan a*sin(2*pi*0.8 t) + tilt 0.6a*sin(2*pi*1.1 t) + damped bumps (pan, 2a, 3 Hz, 0.5 s decay,
  Poisson rate 0.25/s, seed 0); a = 0.1, 0.25, 0.5 deg.
Control: still window (no rotation).
Conditions: base, affine (seed 0), ECC, hybrid, oracle (least-squares similarity fit to the true
inter-frame homography on a 9x9 grid over the window).
3 PTZ telemetry CMC (sweeps only): similarity from reported angles; reading = true angle delayed L frames
  + Gaussian noise sigma; sigma in {0, 0.1, 0.3} deg x L in {0, 1, 3} frames; seed 0.
4 PPE mis-attribution (analysis of tracker outputs, no new runs): each GT person gets a fixed PPE status
  (non-compliant with p = 0.2, seed 0). PPE observations of person p in frame t are credited to the track
  matched to p (IoU > 0.5, Hungarian), i.e. Eq. (1) with the PPE box inside the person box. Track owner =
  GT person matched most often. Metrics: identity mis-attribution = share of credited observations whose
  person != owner; violation error = share whose PPE status != owner's status. Applied to MOT17 real (k=1, 6),
  synthetic pans/shakes, MOT15 and the new PTZ/sway variants.
Metrics: TrackEval HOTA, AssA, IDF1, IDSW, combined over the three sequences. One run (deterministic).
All results reported. No parameter changes after this file.
