"""Strengthening runs B and C (STRENGTHEN_PROTOCOL.md). Time-bounded, resumable.
usage: WORK=... [SEQ_ROOT=...] SEQS=a,b,c LEVELS=pstill0,ptz10,... DETM=yolo|frcnnmap python surv2.py detect|track"""
import sys, os, time, numpy as np, run, common, vptz, cv2
SEQS = os.environ["SEQS"].split(","); LEVELS = os.environ["LEVELS"].split(","); DETM = os.environ.get("DETM", "yolo")
run.STATIC, run.MOVING = SEQS, []
SPECS = [f"{s}_{lv}" for lv in LEVELS for s in SEQS]

def detect_frcnnmap(spec):
    """MOT17 public FRCNN boxes (conf >= 0.25) mapped by H_t; bbox of mapped corners clipped to the window,
    kept if >= 50% inside; appearance features recomputed on the warped frame."""
    out = run.path("dets", spec + ".npy")
    if os.path.exists(out): return
    base = run.parse(spec)[0]; fr, _ = run.frames(spec)
    pub = np.loadtxt(os.path.join(run.SEQ_ROOT, base, "det", "det.txt"), delimiter=",", ndmin=2)
    pub = pub[pub[:, 6] >= 0.25]; enc = common.Encoder(); rows = []
    for i in range(1, fr.n + 1):
        b = pub[pub[:, 0] == i]
        if not len(b): continue
        H = fr.H[i - 1]; boxes, conf = [], []
        for r in b:
            x, y, w, h = r[2:6]
            c = cv2.perspectiveTransform(np.array([[[x, y], [x + w, y], [x, y + h], [x + w, y + h]]], np.float64), H)[0]
            x1, y1 = c.min(0); x2, y2 = c.max(0)
            cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, fr.cw), min(y2, fr.ch)
            if cx2 <= cx1 or cy2 <= cy1: continue
            if (cx2 - cx1) * (cy2 - cy1) < 0.5 * (x2 - x1) * (y2 - y1): continue
            boxes.append([cx1, cy1, cx2 - cx1, cy2 - cy1]); conf.append(r[6])
        if not boxes: continue
        f = enc(fr(i), np.array(boxes))
        for bb, cc, ff in zip(boxes, conf, f):
            rows.append(np.r_[i, -1, bb, cc, -1, -1, -1, ff])
    np.save(out, np.array(rows, np.float32).reshape(-1, 138))

if __name__ == "__main__":
    stage, t0, budget = sys.argv[1], time.time(), float(os.environ.get("BUDGET", 140))
    for sp in SPECS:
        if time.time() - t0 > budget: print("pause"); break
        if stage == "detect":
            if os.path.exists(run.path("dets", sp + ".npy")): continue
            (detect_frcnnmap if DETM == "frcnnmap" else run.detect)(sp); print("det", sp, round(time.time() - t0), flush=True)
        else:
            if not os.path.exists(run.path("dets", sp + ".npy")): continue
            run.write_synth_gt(sp)
            for m in ("base", "cmc", "ecc", "oracle"):
                if not os.path.exists(run.path("trackers", "MOT17SYN-train", "cmc_s0" if m == "cmc" else m, "data", sp + ".txt")): run.track(sp, m, 0)
            print("tracked", sp, round(time.time() - t0), flush=True)
    else:
        print("STAGE DONE")
