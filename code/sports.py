"""SportsMOT real-sweep test (STRENGTHEN_PROTOCOL.md A). usage: WORK=work_sports SEQ_ROOT=~/exp/sports python sports.py detect|track"""
import sys, os, time, run
run.STATIC = []; run.MOVING = open(os.path.join(run.SEQ_ROOT, "SEQS")).read().strip().split(",")
stage, t0, budget = sys.argv[1], time.time(), float(os.environ.get("BUDGET", 140))
for q in run.MOVING:
    if time.time() - t0 > budget: print("pause"); break
    if stage == "detect":
        if os.path.exists(run.path("dets", q + ".npy")): continue
        run.detect(q); print("det", q, round(time.time() - t0), flush=True)
    else:
        if not os.path.exists(run.path("dets", q + ".npy")): continue
        for m in ("base", "cmc", "trans", "ecc", "hybrid"):
            if not os.path.exists(run.path("trackers", "MOT17-train", "cmc_s0" if m == "cmc" else m, "data", q + ".txt")): run.track(q, m, 0)
        print("tracked", q, round(time.time() - t0), flush=True)
else:
    print("STAGE DONE")
