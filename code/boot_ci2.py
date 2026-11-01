"""Merge revision-3 bootstrap shards (work*/boot/<group>_w*.csv) -> ci_rev3.json with 95% percentile CIs of paired differences."""
import csv, glob, json, os, sys, numpy as np
out = {}
for W, tag in [("work", None), ("work_frcnn", "frcnn")]:
    for g in ["full", "stride2", "stride3", "stride6"]:
        fs = glob.glob(f"{W}/boot/{g}_w*.csv")
        if not fs: continue
        rows = {}
        for f in fs:
            for x in csv.DictReader(open(f)):
                rows[int(x["r"]), x["tracker"], x["seq"]] = (float(x["HOTA"]), float(x["IDF1"]))
        R = sorted({k[0] for k in rows}); key = tag if tag else g
        out[f"{key}|n"] = len(R)
        for s in sorted({k[2] for k in rows}):
            for t in ["cmc_s0", "trans", "ecc", "hybrid"]:
                for j, m in enumerate(["HOTA", "IDF1"]):
                    try: d = np.array([rows[r, t, s][j] - rows[r, "base", s][j] for r in R])
                    except KeyError: continue
                    out[f"{key}|{s}|{t}|{m}"] = dict(lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                                                     p_le0=float(np.mean(d <= 0)))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "ci_rev3.json", "w"), indent=1)
print({k: v for k, v in out.items() if k.endswith("|n")})
