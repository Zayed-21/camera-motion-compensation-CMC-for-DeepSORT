import csv, os, run, evaluate
evaluate.link_real_gt()
out = []
for drop in run.MOVING:
    g = [q for q in run.MOVING if q != drop]
    out += evaluate.evaluate("MOT17", g, "moving_minus_" + drop[-2:])
with open(os.path.join(run.WORK, "results_loo.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
