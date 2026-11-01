# Independent moving-camera test — protocol (frozen before any run)
Recorded: 2026-10-31T18:02Z

Sequences (MOT15 train, moving camera, not in MOT17 — byte-checked):
ETH-Bahnhof (1000 fr, 14 fps), ETH-Sunnyday (354 fr, 14 fps), KITTI-13 (340 fr, 10 fps).
Excluded as duplicates of MOT17: ETH-Pedcross2 (=MOT17-05), ADL-Rundle-8 (=MOT17-10),
Venice-2 (=MOT17-02), ADL-Rundle-6 (=MOT17-09). Other MOT15 train sequences are static.

Detector: YOLOv8n (same ONNX, conf 0.25, NMS 0.8, letterbox 640), mars-small128 features. Cached once.
MOT15 public detections (ACF) not used.
Conditions: base, affine (seed 0), translation-only, ECC, hybrid — rules and parameters
unchanged from the MOT17 study (cmc.py, run.py; hybrid = affine if gates pass, else ECC, else identity).
Strides k = 1, 3, 6. Kalman: per-step (main) and time-aware (dt=k, Q*k, max_age=ceil(30/k)) for k>1.
GT: MOT15 gt.txt; rows with flag 0 ignored (MOT15 convention), all others pedestrian, visibility 1.
Metrics: TrackEval HOTA, AssA, IDF1, IDSW (combined over the 3 sequences + per sequence).
Uncertainty: same circular moving-block bootstrap (60 processed frames, 1000 resamples, seed 2026).
All results reported, including negative ones. No parameter changes after this file.

## Addendum — second detector (recorded 2026-10-31T22:24Z, before running it)
MOT15 provides only ACF public detections (no FRCNN). Second-detector runs use these ACF boxes as provided
(already thresholded by the benchmark; no confidence threshold applied, no extra suppression), with the same
appearance network, tracker settings, estimators, strides and Kalman variants. Results reported whatever they show.
