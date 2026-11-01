import time, os
import run
t0=time.time()
def have(sp, name): return os.path.exists(run.path("trackers","MOT17SYN-train",name,"data",sp+".txt"))
names=("base","cmc_s0","oracle","trans","ecc")
for sp in run.synth_specs():
    if all(have(sp,n) for n in names): continue
    if time.time()-t0>120: print("pause"); break
    run.write_synth_gt(sp)
    if not have(sp,"base"): run.track(sp,"base")
    if not have(sp,"cmc_s0"): run.estimate_cmc(sp,0); run.track(sp,"cmc",0)
    if not have(sp,"oracle"): run.track(sp,"oracle")
    for k in ("trans","ecc"):
        if not have(sp,k): run.estimate_alt(sp,k); run.track(sp,k)
    print("done", sp, round(time.time()-t0), flush=True)
else:
    print("ALL SYNTH DONE")
