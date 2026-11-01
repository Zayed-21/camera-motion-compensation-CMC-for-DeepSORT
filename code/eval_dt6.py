import csv, os, run, evaluate
out = evaluate.evaluate("MOT17STR", [f"{q}_stride6" for q in run.MOVING], "moving_stride6")
out += evaluate.evaluate("MOT17STR", [f"{q}_stride6" for q in run.MOVING])
with open(os.path.join(run.WORK, "results_dt.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
