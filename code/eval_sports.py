import csv, os, run, evaluate
run.STATIC = []; run.MOVING = open(os.path.join(run.SEQ_ROOT, "SEQS")).read().strip().split(",")
evaluate.link_real_gt()
out = evaluate.evaluate("MOT17", run.MOVING, "sports") + evaluate.evaluate("MOT17", run.MOVING)
with open(os.path.join(run.WORK, "results_sports.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out))
