import sys, csv, os
import run, evaluate
out = []
for k in (2, 3, 6):
    out += evaluate.evaluate("MOT17STR", [f"{q}_stride{k}" for q in run.MOVING], f"moving_stride{k}")
    out += [r for r in evaluate.evaluate("MOT17STR", [f"{q}_stride{k}" for q in run.MOVING]) ]
with open(os.path.join(run.WORK, "results_dt.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out))
