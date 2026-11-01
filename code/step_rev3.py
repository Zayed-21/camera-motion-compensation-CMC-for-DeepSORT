"""Revision 3 driver (resumable, time-boxed): tracks every missing condition on real and stride specs.
YOLOv8n: `python step_rev3.py hybrid`  (WORK=work).  FRCNN: `DET=frcnn WORK=work_frcnn python step_rev3.py all`."""
import os, sys, time
import run
BUDGET = float(os.environ.get("BUDGET", 150)); t0 = time.time()
MODES = {"hybrid": ["hybrid"], "all": ["base", "cmc_s0", "trans", "ecc", "hybrid"]}[sys.argv[1]]
def have(sp, name):
    bench = "MOT17-train" if run.parse(sp)[1] is None else "MOT17STR-train"
    return os.path.exists(run.path("trackers", bench, name, "data", sp + ".txt"))
specs = run.STATIC + run.MOVING + run.stride_specs()
for sp in specs:
    if all(have(sp, m) for m in MODES): continue
    if time.time() - t0 > BUDGET: print("pause"); break
    if run.parse(sp)[1] == "stride": run.write_stride_gt(sp)
    run.detect(sp)
    for m in MODES:
        if have(sp, m): continue
        if m == "base": run.track(sp, "base")
        elif m == "cmc_s0": run.estimate_cmc(sp, 0); run.track(sp, "cmc", 0)
        elif m in ("trans", "ecc"): run.estimate_alt(sp, m); run.track(sp, m)
        else: run.estimate_cmc(sp, 0); run.estimate_alt(sp, "ecc"); run.track(sp, "hybrid")
    print("done", sp, round(time.time() - t0), flush=True)
else:
    print("ALL DONE")
