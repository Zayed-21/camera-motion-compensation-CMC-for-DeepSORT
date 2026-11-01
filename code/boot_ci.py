"""Merge bootstrap shards and write 95% percentile CIs of paired differences to work/ci.json."""
import csv, glob, json, os, numpy as np
import run
rows = {}
for f in sorted(glob.glob(os.path.join(run.WORK, "bootstrap*.csv"))):
    for x in csv.DictReader(open(f)):
        rows[int(x["r"]), x["tracker"], x["seq"]] = (float(x["HOTA"]), float(x["IDF1"]))
R = sorted({k[0] for k in rows}); seqs = sorted({k[2] for k in rows})
out = {"resamples": len(R), "missing": sorted(set(range(1000)) - set(R))}
for s in seqs:
    for t in ["cmc_s0", "trans", "ecc"]:
        for j, m in enumerate(["HOTA", "IDF1"]):
            d = np.array([rows[r, t, s][j] - rows[r, "base", s][j] for r in R])
            out[f"{s}|{t}|{m}"] = dict(lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                       med=float(np.median(d)), p_le0=float(np.mean(d <= 0)))
json.dump(out, open(os.path.join(run.WORK, "ci.json"), "w"), indent=1)
print(out["resamples"], out["missing"][:5])
for k, v in out.items():
    if "|" in k: print(k, f"[{v['lo']:+.2f}, {v['hi']:+.2f}] med {v['med']:+.2f} P(d<=0) {v['p_le0']:.3f}")
