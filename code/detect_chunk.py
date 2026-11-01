"""Resumable version of run.detect() for machines that limit how long one command may run.
Processes frames of <spec> until a time budget is used, keeps partial results in
work/dets/<spec>.part.npz, and writes the same work/dets/<spec>.npy/.json as run.detect()."""
import json, os, sys, time
import numpy as np
import run, common

def step(spec, budget=150.0):
    out = run.path("dets", spec + ".npy")
    if os.path.exists(out):
        return True
    part = run.path("dets", spec + ".part.npz")
    fr, _ = run.frames(spec)
    if os.path.exists(part):
        z = np.load(part); rows = list(z["rows"]); t_det = list(z["t_det"]); t_enc = list(z["t_enc"]); nxt = int(z["next"])
    else:
        rows, t_det, t_enc, nxt = [], [], [], 1
    det, enc = common.PersonDetector(), common.Encoder()
    t_start = time.time()
    i = nxt
    while i <= fr.n and time.time() - t_start < budget:
        img = fr(i)
        t0 = time.perf_counter(); b = det(img); t1 = time.perf_counter()
        f = enc(img, b[:, :4]); t2 = time.perf_counter()
        t_det.append(t1 - t0); t_enc.append(t2 - t1)
        for bb, ff in zip(b, f):
            rows.append(np.r_[i, -1, bb[:4], bb[4], -1, -1, -1, ff].astype(np.float32))
        i += 1
    if i > fr.n:
        m = np.array(rows, np.float32).reshape(-1, 138)
        np.save(out, m)
        json.dump(dict(det_ms=1e3 * float(np.median(t_det)), enc_ms=1e3 * float(np.median(t_enc)),
                       frames=fr.n, dets=len(m)), open(run.path("dets", spec + ".json"), "w"))
        if os.path.exists(part): os.remove(part)
        return True
    np.savez(part, rows=np.array(rows, np.float32).reshape(-1, 138), t_det=t_det, t_enc=t_enc, next=i)
    print(spec, "frame", i, "of", fr.n, flush=True)
    return False

if __name__ == "__main__":
    budget = float(os.environ.get("BUDGET", 150))
    t0 = time.time()
    for spec in sys.argv[1:]:
        left = budget - (time.time() - t0)
        if left < 10: break
        if not step(spec, left): break
        print(spec, "done", flush=True)
