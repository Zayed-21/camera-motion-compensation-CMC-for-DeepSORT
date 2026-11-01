"""Revision analyses (B1-B3, C2/C3 summaries, rerun-vs-original). Writes work/revision.json."""
import os, csv, json, numpy as np
import run, synth, common
ORIG = os.environ.get("ORIG", os.path.expanduser("~/mnt/refs/results/results.csv"))
M = ["HOTA", "DetA", "AssA", "IDF1", "IDSW", "MOTA", "Frag"]
def table(f):
    return {(r["tracker"], r["seq"]): {k: float(r[k]) for k in M} for r in csv.DictReader(open(f))}
orig, new = table(ORIG), table(os.path.join(run.WORK, "results.csv"))
out = {}
real = run.STATIC + run.MOVING + ["COMBINED_static", "COMBINED_moving", "COMBINED_all"]
# B1: per-sequence detection/CLEAR metrics, original run
out["B1"] = {s: {k: [orig["base", s][k], orig["cmc_s0", s][k] - orig["base", s][k]] for k in M} for s in real}
# B2: ground-truth inventory (pedestrian class 1, consider flag 1)
inv = {}
for s in run.STATIC + run.MOVING:
    g = np.loadtxt(os.path.join(run.SEQ_ROOT, s, "gt", "gt.txt"), delimiter=",")
    p = g[(g[:, 6] == 1) & (g[:, 7] == 1)]
    n = common.seqinfo(os.path.join(run.SEQ_ROOT, s))["n"]
    inv[s] = dict(boxes=int(len(p)), ids=int(len(np.unique(p[:, 1]))), per_frame=len(p) / n,
                  lowvis=100 * float(np.mean(p[:, 8] < 0.5)), med_h=float(np.median(p[:, 5])))
out["B2"] = inv
# B3 + C2/C3: estimator error against the true synthetic offset (pixels, non-fallback frames)
est = {}
for kind, mag in synth.LEVELS:
    if kind == "still": continue
    lvl = f"{kind}{mag}"; e = {}
    for tag in ["s0", "trans", "ecc"]:
        errs, fbs = [], []
        for s in run.STATIC:
            spec = f"{s}_{lvl}"; _, offs = run.frames(spec)
            z = np.load(os.path.join(run.WORK, "cmc", f"{spec}_{tag}.npz"))
            true_t = -(offs[1:] - offs[:-1])          # image content moves opposite to the window
            fb = z["info"][1:, 2] > 0
            err = np.linalg.norm(z["A"][1:, :, 2] - true_t, axis=1)
            errs.append(err[~fb]); fbs.append(fb)
        errs = np.concatenate(errs); fbs = np.concatenate(fbs)
        e[tag] = dict(mean=float(errs.mean()), p95=float(np.percentile(errs, 95)), fb=100 * float(fbs.mean()))
    c = f"COMBINED_{lvl}"
    e["gap"] = {t: {k: new[t, c][k] - new["oracle", c][k] for k in ["HOTA", "IDF1"]} for t in ["cmc_s0", "trans", "ecc"]}
    e["gap_orig"] = {k: orig["cmc_s0", c][k] - orig["oracle", c][k] for k in ["HOTA", "IDF1"]}
    est[lvl] = e
out["B3"] = est
# rerun tables
T = ["base", "cmc_s0", "trans", "ecc", "oracle"]
out["rerun"] = {s: {t: {k: new[t, s][k] for k in ["HOTA", "AssA", "IDF1", "IDSW"]} for t in T if (t, s) in new}
                for s in sorted({s for _, s in new})}
out["orig"] = {s: {t: {k: orig[t, s][k] for k in ["HOTA", "AssA", "IDF1", "IDSW"]} for t in ["base", "cmc_s0", "oracle"] if (t, s) in orig}
               for s in sorted({s for _, s in orig})}
# estimator fallback / time on real and stride sequences
fbt = {}
for s in run.STATIC + run.MOVING:
    for sp in [s] + [f"{s}_stride{k}" for k in run.STRIDES]:
        fbt[sp] = {}
        for tag in ["s0", "trans", "ecc"]:
            z = np.load(os.path.join(run.WORK, "cmc", f"{sp}_{tag}.npz"))["info"][1:]
            fbt[sp][tag] = dict(fb=100 * float(z[:, 2].mean()), ms=float(np.median(z[:, 3])))
out["est_real"] = fbt
json.dump(out, open(os.path.join(run.WORK, "revision.json"), "w"), indent=1)
print("ok")
