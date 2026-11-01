"""Revision 4: merge bootstrap shards for time-aware and MOT15 runs -> ci_rev4.json (95% percentile CIs of paired diffs)."""
import csv, glob, json, numpy as np
out = {}
for W in ["work_dt", "work_frcnn_dt", "work_mot15", "work_mot15_dt", "work_mot15_acf", "work_surv", "work_sports", "work_m15ptz", "work_surv_frcnn"]:
    for g in ["full", "stride2", "stride3", "stride6"] + [f"syn_{x}" for x in ("ptz5","ptz10","ptz20","ptz40","sway10","sway25","sway50")]:
        fs = glob.glob(f"{W}/boot/{g}_w*.csv")
        if not fs: continue
        rows = {}
        for f in fs:
            for x in csv.DictReader(open(f)):
                rows[int(x["r"]), x["tracker"], x["seq"]] = (float(x["HOTA"]), float(x["IDF1"]))
        R = sorted({k[0] for k in rows})
        if len(R) < 1000: continue
        for s in sorted({k[2] for k in rows}):
            for t in ["cmc_s0", "trans", "ecc", "hybrid"]:
                for j, m in enumerate(["HOTA", "IDF1"]):
                    try: d = np.array([rows[r, t, s][j] - rows[r, "base", s][j] for r in R])
                    except KeyError: continue
                    out[f"{W}|{g}|{s}|{t}|{m}"] = [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]
json.dump(out, open("ci_rev4.json", "w"), indent=0)
for k, v in out.items():
    if "COMBINED" in k: print(k, v)
