import csv, os, run, evaluate
LV = [("pstill", 0), ("ptz", 5), ("ptz", 10), ("ptz", 20), ("ptz", 40), ("sway", 10), ("sway", 25), ("sway", 50)]
out = []
for k, m in LV:
    out += evaluate.evaluate("MOT17SYN", [f"{s}_{k}{m}" for s in run.STATIC], f"{k}{m}")
with open(os.path.join(run.WORK, "results_surv.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out))
