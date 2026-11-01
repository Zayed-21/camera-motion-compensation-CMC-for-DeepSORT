# Strengthening experiments for the surveillance direction — protocol (frozen before any run)
Recorded: 2026-11-01T12:25Z

## A. Real sweeping camera: SportsMOT val (CC BY-NC 4.0)
Selection rule (independent of results): the first clip (lowest clip number) of each of the 13 source videos
in the val split -> 13 clips, 8,074 frames, 1280x720, 25 fps:
- v_00HRwkvvjtQ_c001 (1162 frames)
- v_0kUtTtmLaJA_c004 (542 frames)
- v_2QhNRucNC7E_c017 (450 frames)
- v_4-EmEtrturE_c009 (250 frames)
- v_4r8QL_wglzQ_c001 (886 frames)
- v_5ekaksddqrc_c001 (550 frames)
- v_9MHDmAMxO5I_c002 (239 frames)
- v_BgwzTUxJaeU_c008 (500 frames)
- v_G-vNjfx1GGc_c004 (675 frames)
- v_ITo3sCnpw_k_c007 (701 frames)
- v_cC2mHWqMcjk_c007 (681 frames)
- v_dw7LOz17Omg_c053 (550 frames)
- v_i2_L4qquVg0_c006 (888 frames)
Detector: YOLOv8n (same settings), mars-small128 features, cached once. No public detections exist.
Conditions: base, affine (seed 0), translation-only, ECC, hybrid; full frame rate only; per-step Kalman (main settings).
Camera motion per clip: median affine inter-frame translation (% width) and median rotation/scale change, reported.
Metrics: TrackEval HOTA, AssA, IDF1, IDSW (combined over 13 clips + per clip); block bootstrap as before.
Domain caveat stated in paper: athletes in uniforms (appearance less discriminative) — not workers.

## B. Second detector on the virtual PTZ camera
MOT17 public FRCNN boxes of MOT17-02/-04/-09 (confidence >= 0.25, as in the main study) mapped by the same
homography H_t as the ground truth; box = bounding box of mapped corners clipped to the window; kept if >= 50%
of the mapped box lies inside the window. Appearance features recomputed on the warped frames.
Levels: still, PTZ sweep 5/10/20/40 deg/s, sway 0.10/0.25/0.50 deg (same as the YOLOv8n runs).
Conditions: base, affine, ECC, oracle. Metrics and bootstrap (affine vs base) as before.

## C. More scenes for the virtual PTZ camera
All static-camera MOT15 training sequences that are not duplicates of MOT17: PETS09-S2L1 (768x576, 7 fps),
TUD-Stadtmitte (640x480, 25 fps), TUD-Campus (640x480, 25 fps), KITTI-17 (1224x370, 10 fps).
Virtual camera scaled to each frame: focal length 1662.8*w/1920 (same 60 deg HFOV), window 2/3 of width and height;
identical for MOT17 (1280x720 of 1920x1080). Same sweep/sway definitions in deg and deg/s (so per-frame motion is
larger at low frame rate; reported). Levels: still, sweep 10/20/40 deg/s, sway 0.50 deg. YOLOv8n; base, affine, ECC, oracle.
MOT15 GT rows flagged zero ignored. Combined over the 4 sequences; bootstrap affine vs base.

All results reported. No parameter changes after this file.
