import sys, time, os
import run
t0=time.time()
def have(sp, name): return os.path.exists(run.path("trackers","MOT17STR-train",name,"data",sp+".txt"))
for sp in run.stride_specs():
    if all(have(sp,n) for n in ("base","cmc_s0","trans","ecc")): continue
    if time.time()-t0>110: print("pause"); break
    run.write_stride_gt(sp)
    if not have(sp,"base"): run.track(sp,"base")
    if not have(sp,"cmc_s0"): run.estimate_cmc(sp,0); run.track(sp,"cmc",0)
    for k in ("trans","ecc"):
        if not have(sp,k): run.estimate_alt(sp,k); run.track(sp,k)
    print("done", sp, round(time.time()-t0), flush=True)
else:
    print("ALL STRIDES DONE")
