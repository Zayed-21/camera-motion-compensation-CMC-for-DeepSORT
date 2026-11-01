import csv, os, run, evaluate
SEQS = os.environ["SEQS"].split(","); LEVELS = os.environ["LEVELS"].split(",")
out = []
for lv in LEVELS:
    out += evaluate.evaluate("MOT17SYN", [f"{s}_{lv}" for s in SEQS], lv)
with open(os.path.join(run.WORK, "results_surv2.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
