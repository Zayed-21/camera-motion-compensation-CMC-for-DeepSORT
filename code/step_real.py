import sys, time, os
import run
t0=time.time()
for s in sys.argv[1:]:
    if time.time()-t0>120: print("stop before", s); break
    if not os.path.exists(run.path("trackers","MOT17-train","base","data",s+".txt")): run.track(s,"base")
    run.estimate_cmc(s,0)
    if not os.path.exists(run.path("trackers","MOT17-train","cmc_s0","data",s+".txt")): run.track(s,"cmc",0)
    print("done", s, round(time.time()-t0), flush=True)
