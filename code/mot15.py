"""Independent moving-camera test on MOT15 (protocol frozen in MOT15_PROTOCOL.md).
usage: WORK=work_mot15 SEQ_ROOT=~/exp/mot15 python mot15.py <stage>   stage: detect | track | dt"""
import sys, os, math, time
import run
run.STATIC = []; run.MOVING = ["ETH-Bahnhof", "ETH-Sunnyday", "KITTI-13"]
stage = sys.argv[1]; t0 = time.time()
if os.environ.get("DET") == "frcnn": run.DS["min_confidence"] = -1e9   # MOT15 ACF public boxes, used as provided
def have(bench, name, sp): return os.path.exists(run.path("trackers", bench, name, "data", sp + ".txt"))
if stage == "detect":
    for q in run.MOVING: run.detect(q); print("det", q, round(time.time() - t0), flush=True)
elif stage == "track":
    for q in run.MOVING:
        for k in (1, 3, 6):
            sp = q if k == 1 else f"{q}_stride{k}"
            bench = "MOT17-train" if k == 1 else "MOT17STR-train"
            if k > 1: run.write_stride_gt(sp)
            for m in ("base", "cmc", "trans", "ecc", "hybrid"):
                if not have(bench, "cmc_s0" if m == "cmc" else m, sp): run.track(sp, m, 0)
            print("tracked", sp, round(time.time() - t0), flush=True)
