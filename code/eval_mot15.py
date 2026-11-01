import csv, os, sys, run, evaluate
run.STATIC, run.MOVING = [], ["ETH-Bahnhof", "ETH-Sunnyday", "KITTI-13"]
out = []
if os.path.isdir(os.path.join(run.WORK, "trackers", "MOT17-train")):
    evaluate.link_real_gt()
    out += evaluate.evaluate("MOT17", run.MOVING, "mot15") + evaluate.evaluate("MOT17", run.MOVING)
for k in (3, 6):
    g = [f"{q}_stride{k}" for q in run.MOVING]
    out += evaluate.evaluate("MOT17STR", g, f"mot15_stride{k}") + evaluate.evaluate("MOT17STR", g)
with open(os.path.join(run.WORK, "results_mot15.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out))
