import os, csv, sys
import evaluate, run
evaluate.link_real_gt()
rows = evaluate.evaluate("MOT17")
for tag, g in (("static", run.STATIC), ("moving", run.MOVING), ("all", run.STATIC + run.MOVING)):
    rows += evaluate.evaluate("MOT17", g, tag)
out = sys.argv[1] if len(sys.argv) > 1 else "work/results_real.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
