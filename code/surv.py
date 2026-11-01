"""Surveillance-motion experiments driver (time-bounded, resumable). usage: WORK=work_surv python surv.py detect|track"""
import sys, os, time, run
LV = [("pstill", 0), ("ptz", 5), ("ptz", 10), ("ptz", 20), ("ptz", 40), ("sway", 10), ("sway", 25), ("sway", 50)]
SPECS = [f"{s}_{k}{m}" for k, m in LV for s in run.STATIC]
TEL = [f"tel{s}_{l}" for s in (0, 1, 3) for l in (0, 1, 3)]
if __name__ == "__main__":
    stage, t0, budget = sys.argv[1], time.time(), float(os.environ.get("BUDGET", 150))
    def have(name, sp): return os.path.exists(run.path("trackers", "MOT17SYN-train", name, "data", sp + ".txt"))
    for sp in SPECS:
        if time.time() - t0 > budget: print("pause"); break
        if stage == "detect":
            if os.path.exists(run.path("dets", sp + ".npy")): continue
            run.detect(sp); print("det", sp, round(time.time() - t0), flush=True)
        else:
            if not os.path.exists(run.path("dets", sp + ".npy")): continue
            run.write_synth_gt(sp)
            modes = ["base", "cmc", "ecc", "hybrid", "oracle"] + (TEL if "_ptz" in sp else [])
            for m in modes:
                if not have("cmc_s0" if m == "cmc" else m, sp): run.track(sp, m, 0)
            print("tracked", sp, round(time.time() - t0), flush=True)
    else:
        print("STAGE DONE")
