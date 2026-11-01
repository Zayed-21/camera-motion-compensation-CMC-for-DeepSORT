"""Experiment driver. Stages are cached under work/ so each can be re-run independently.

  python run.py detect  <spec>          # YOLOv8 + appearance features, once per frame
  python run.py cmc     <spec> <seed>   # camera-motion estimates for every frame
  python run.py track   <spec> <mode> [seed]   # mode: base | cmc | oracle
  python run.py all                      # everything in the protocol
  python run.py eval                     # TrackEval on every result folder

<spec> is a MOT17 sequence name (MOT17-02) or a synthetic variant (MOT17-02_pan10)."""
import json, os, sys, time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "deep_sort"))
from deep_sort import nn_matching                      # noqa: E402
from deep_sort.detection import Detection              # noqa: E402
from deep_sort.tracker import Tracker                  # noqa: E402
import common, cmc, synth, vptz                         # noqa: E402

WORK = os.environ.get("WORK", os.path.join(ROOT, "work"))
DET = os.environ.get("DET", "yolo")   # "frcnn": MOT17 public FRCNN detections (real sequences only)
SEQ_ROOT = os.environ.get("SEQ_ROOT", os.path.join(common.DATA, "MOT17", "train"))
STATIC = ["MOT17-02", "MOT17-04", "MOT17-09"]
MOVING = ["MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"]
SEEDS = [0, 1, 2, 3, 4]
# DeepSORT settings: reference defaults, nn_budget from the repository README example;
# min_confidence equals the detector threshold so no extra filtering; nms 1.0 = off
# (the detector already applies NMS, IoU 0.8 as in the capstone's inference code).
DS = dict(min_confidence=0.25, nms_max_overlap=1.0, max_cosine_distance=0.2, nn_budget=100,
          max_iou_distance=0.7, max_age=30, n_init=3)

def parse(spec):
    base, _, var = spec.partition("_")
    if not var:
        return base, None, None
    kind = var.rstrip("0123456789"); mag = int(var[len(kind):] or 0)
    return base, kind, mag

def frames(spec):
    base, kind, mag = parse(spec)
    d = os.path.join(SEQ_ROOT, base)
    if kind is None:
        return common.Frames(d), None
    if kind == "stride":                     # frame subsampling: process every mag-th frame
        return common.Frames(d, step=mag), None
    if kind in vptz.KINDS:                   # virtual PTZ / sway camera (rotation homography)
        fr = vptz.Frames(d, kind, mag, common.seqinfo); return fr, fr
    info = common.seqinfo(d)
    offs, crop = synth.offsets(kind, mag, info["n"], info["w"], info["h"], seed=0)
    return common.Frames(d, offs, crop), offs

def path(*p):
    f = os.path.join(WORK, *p); os.makedirs(os.path.dirname(f), exist_ok=True); return f

# ---------------- stages ----------------
def detect(spec):
    out = path("dets", spec + ".npy")
    if os.path.exists(out):
        return np.load(out)
    base, kind, mag = parse(spec)
    if kind == "stride":                     # same detections as the full sequence, kept frames renumbered
        m = detect(base).copy(); keep = (m[:, 0] - 1) % mag == 0
        m = m[keep]; m[:, 0] = (m[:, 0] - 1) // mag + 1
        np.save(out, m); return m
    if DET == "frcnn":
        return detect_public(spec)
    fr, _ = frames(spec)
    det, enc = common.PersonDetector(), common.Encoder()
    rows, t_det, t_enc = [], [], []
    for i in range(1, fr.n + 1):
        img = fr(i)
        t0 = time.perf_counter(); b = det(img); t1 = time.perf_counter()
        f = enc(img, b[:, :4]); t2 = time.perf_counter()
        t_det.append(t1 - t0); t_enc.append(t2 - t1)
        for bb, ff in zip(b, f):
            rows.append(np.r_[i, -1, bb[:4], bb[4], -1, -1, -1, ff])
    m = np.array(rows, np.float32).reshape(-1, 138)
    np.save(out, m)
    json.dump(dict(det_ms=1e3 * float(np.median(t_det)),
                   enc_ms=1e3 * float(np.median(t_enc)), frames=fr.n, dets=len(m)),
              open(path("dets", spec + ".json"), "w"))
    return m

def detect_public(spec):
    """MOT17 public FRCNN boxes (frame, -1, x, y, w, h, conf) + appearance features, same row layout as detect()."""
    assert parse(spec)[1] is None, "public detections exist only for the real sequences"
    out = path("dets", spec + ".npy")
    pub = np.loadtxt(os.path.join(SEQ_ROOT, spec, "det", "det.txt"), delimiter=",", ndmin=2)
    fr, _ = frames(spec); enc = common.Encoder(); rows = []
    for i in range(1, fr.n + 1):
        b = pub[pub[:, 0] == i]
        if not len(b): continue
        f = enc(fr(i), b[:, 2:6])
        for bb, ff in zip(b, f):
            rows.append(np.r_[i, -1, bb[2:6], bb[6], -1, -1, -1, ff])
    m = np.array(rows, np.float32).reshape(-1, 138)
    np.save(out, m); return m

def estimate_cmc(spec, seed):
    """Camera-motion estimates for all seeds in one pass: corner detection and optical flow
    are deterministic and shared; only RANSAC is re-run per seed (cv2.setRNGSeed per frame).
    info columns: n_tracked, inlier_ratio, fallback, ms (shared flow time + this seed's RANSAC)."""
    out = path("cmc", f"{spec}_s{seed}.npz")
    if os.path.exists(out):
        return np.load(out)
    dets = detect(spec)
    fr, _ = frames(spec)
    A = np.tile(np.eye(2, 3), (len(SEEDS), fr.n, 1, 1)); info = np.zeros((len(SEEDS), fr.n, 4))
    prev = cmc.to_gray(fr(1))
    for i in range(2, fr.n + 1):
        cur = cmc.to_gray(fr(i))
        boxes = dets[dets[:, 0] == i - 1, 2:6]
        t0 = time.perf_counter(); pts = cmc.track_points(prev, cur, boxes); t_flow = time.perf_counter() - t0
        for k, sd in enumerate(SEEDS):
            cv2.setRNGSeed(sd * 1_000_003 + i)
            t1 = time.perf_counter(); A[k, i - 1], n, r, fb = cmc.fit(pts)
            info[k, i - 1] = (n, r, fb, 1e3 * (t_flow + time.perf_counter() - t1))
        prev = cur
    for k, sd in enumerate(SEEDS):
        np.savez(path("cmc", f"{spec}_s{sd}.npz"), A=A[k], info=info[k])
    return np.load(out)

def estimate_alt(spec, kind):
    """Camera-motion estimates for the revision's extra conditions: kind 'trans' (translation-only
    ablation of the CMC step) or 'ecc' (StrongSORT-style ECC reference). Deterministic, one run.
    info columns as in estimate_cmc: n, ratio (or ECC correlation), fallback, ms."""
    out = path("cmc", f"{spec}_{kind}.npz")
    if os.path.exists(out):
        return np.load(out)
    dets = detect(spec)
    fr, _ = frames(spec)
    A = np.tile(np.eye(2, 3), (fr.n, 1, 1)); info = np.zeros((fr.n, 4))
    prev_img = fr(1); prev = cmc.to_gray(prev_img)
    for i in range(2, fr.n + 1):
        img = fr(i)
        t0 = time.perf_counter()
        if kind == "trans":
            cur = cmc.to_gray(img)
            pts = cmc.track_points(prev, cur, dets[dets[:, 0] == i - 1, 2:6])
            A[i - 1], n, r, fb = cmc.fit_translation(pts)
            prev = cur
        else:
            A[i - 1], n, r, fb = cmc.estimate_ecc(prev_img, img)
        info[i - 1] = (n, r, fb, 1e3 * (time.perf_counter() - t0))
        prev_img = img
    np.savez(out, A=A, info=info)
    return np.load(out)

def track(spec, mode, seed=0, write=True):
    """Runs DeepSORT; mode 'cmc' warps predictions with estimated A, 'oracle' with the true one."""
    dets = detect(spec)
    fr, offs = frames(spec)
    if mode == "cmc":
        A = estimate_cmc(spec, seed)["A"]
    elif mode in ("trans", "ecc"):
        A = estimate_alt(spec, mode)["A"]
    elif mode == "hybrid":   # affine where it passed its gates, ECC on affine fallback frames
        a, e = estimate_cmc(spec, 0), estimate_alt(spec, "ecc")
        A = np.where((a["info"][:, 2] > 0)[:, None, None], e["A"], a["A"])
    elif mode == "oracle" and isinstance(offs, vptz.Frames):
        A = offs.oracle()
    elif mode.startswith("tel"):             # PTZ telemetry: tel<sigma x10>_<latency frames>
        sg, lat = mode[3:].split("_"); A = offs.telemetry(int(sg) / 10, int(lat))
    elif mode == "oracle":
        assert offs is not None, "oracle needs a synthetic sequence"
        A = np.stack([cmc.oracle(offs, i) for i in range(1, fr.n + 1)])
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", DS["max_cosine_distance"], DS["nn_budget"])
    tracker = Tracker(metric, max_iou_distance=DS["max_iou_distance"], max_age=DS["max_age"], n_init=DS["n_init"])
    frame_idx = dets[:, 0].astype(int)
    results, t_upd = [], []
    for i in range(1, fr.n + 1):
        rows = dets[frame_idx == i]
        ds = [Detection(r[2:6], r[6], r[10:]) for r in rows if r[6] >= DS["min_confidence"]]
        t0 = time.perf_counter()
        tracker.predict()
        if mode in ("cmc", "oracle", "trans", "ecc", "hybrid") or mode.startswith("tel"):
            cmc.warp_tracks(tracker.tracks, A[i - 1])
        tracker.update(ds)
        t_upd.append(time.perf_counter() - t0)
        for tr in tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update > 1:
                continue
            x, y, w, h = tr.to_tlwh()
            results.append(f"{i},{tr.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    if write:
        name = f"{mode}_s{seed}" if mode == "cmc" else mode
        kind = parse(spec)[1]
        bench = "MOT17-train" if kind is None else "MOT17STR-train" if kind == "stride" else "MOT17SYN-train"
        with open(path("trackers", bench, name, "data", spec + ".txt"), "w") as f:
            f.write("\n".join(results) + ("\n" if results else ""))
    return np.array(t_upd)

# ---------------- ground truth for synthetic sequences ----------------
def write_synth_gt(spec):
    base, kind, mag = parse(spec)
    d = os.path.join(SEQ_ROOT, base); info = common.seqinfo(d)
    if kind in vptz.KINDS:
        gt = np.loadtxt(os.path.join(d, "gt", "gt.txt"), delimiter=",")
        fr_ = frames(spec)[0]; g = fr_.gt(gt); out = path("gt", "MOT17SYN-train", spec, "gt", "gt.txt")
        np.savetxt(out, g, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.6f")
        with open(path("gt", "MOT17SYN-train", spec, "seqinfo.ini"), "w") as f:
            f.write(f"[Sequence]\nname={spec}\nimDir=img1\nframeRate={info['fps']:g}\n"
                    f"seqLength={info['n']}\nimWidth={fr_.cw}\nimHeight={fr_.ch}\nimExt=.jpg\n")
        return
    offs, (cw, ch) = synth.offsets(kind, mag, info["n"], info["w"], info["h"], seed=0)
    gt = np.loadtxt(os.path.join(d, "gt", "gt.txt"), delimiter=",")
    g = synth.shift_gt(gt, offs, (cw, ch))
    out = path("gt", "MOT17SYN-train", spec, "gt", "gt.txt")
    np.savetxt(out, g, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.6f")
    with open(path("gt", "MOT17SYN-train", spec, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\nname={spec}\nimDir=img1\nframeRate={info['fps']:g}\n"
                f"seqLength={info['n']}\nimWidth={cw}\nimHeight={ch}\nimExt=.jpg\n")

STRIDES = [2, 3, 6]

def write_stride_gt(spec):
    base, _, s = parse(spec)
    d = os.path.join(SEQ_ROOT, base); info = common.seqinfo(d)
    gt = np.loadtxt(os.path.join(d, "gt", "gt.txt"), delimiter=",")
    gt = gt[(gt[:, 0] - 1) % s == 0]; gt[:, 0] = (gt[:, 0] - 1) // s + 1
    np.savetxt(path("gt", "MOT17STR-train", spec, "gt", "gt.txt"), gt, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.6f")
    with open(path("gt", "MOT17STR-train", spec, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\nname={spec}\nimDir=img1\nframeRate={info['fps'] / s:g}\n"
                f"seqLength={-(-info['n'] // s)}\nimWidth={info['w']}\nimHeight={info['h']}\nimExt=.jpg\n")

def stride_specs():
    return [f"{q}_stride{s}" for q in STATIC + MOVING for s in STRIDES]

def stride_protocol():
    for sp in stride_specs():
        write_stride_gt(sp); track(sp, "base")
        for seed in SEEDS:
            estimate_cmc(sp, seed); track(sp, "cmc", seed)
        print("tracked", sp, flush=True)

def synth_specs():
    return [f"{s}_{k}{m}" for s in STATIC for k, m in synth.LEVELS]

def protocol():
    real = STATIC + MOVING
    for s in real + synth_specs():
        t = time.time(); detect(s); print("detect", s, round(time.time() - t), "s", flush=True)
    for s in synth_specs():
        write_synth_gt(s)
    for s in real + synth_specs():
        track(s, "base")
        for seed in SEEDS:
            estimate_cmc(s, seed); track(s, "cmc", seed)
        if parse(s)[1] is not None:
            track(s, "oracle")
        print("tracked", s, flush=True)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "detect": detect(sys.argv[2])
    elif cmd == "cmc": estimate_cmc(sys.argv[2], int(sys.argv[3]))
    elif cmd == "track": track(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    elif cmd == "all": protocol()
    elif cmd == "stride": stride_protocol()
    elif cmd == "alt":                       # revision: translation-only and ECC conditions
        specs = sys.argv[2:] or (STATIC + MOVING + stride_specs() + synth_specs())
        for sp in specs:
            for kind in ("trans", "ecc"):
                estimate_alt(sp, kind); track(sp, kind)
            print("alt", sp, flush=True)
    else: raise SystemExit(__doc__)
