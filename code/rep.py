import os, sys, filecmp, numpy as np, run
sp = sys.argv[1]; O = "/sessions/rcw-0119jsvkrnpwuc2uanmnp1td/exp/code/work"
for m in ("base", "cmc", "ecc"):
    run.track(sp, m, 0)
    n = "cmc_s0" if m == "cmc" else m
    a = os.path.join(run.WORK, "trackers/MOT17SYN-train", n, "data", sp + ".txt"); b = os.path.join(O, "trackers/MOT17SYN-train", n, "data", sp + ".txt")
    print(sp, m, "tracker identical:", filecmp.cmp(a, b, shallow=False) if os.path.exists(b) else "no ref")
for f in (f"{sp}_s0.npz", f"{sp}_ecc.npz"):
    p, q = os.path.join(run.WORK, "cmc", f), os.path.join(O, "cmc", f)
    if os.path.exists(q): print(f, "max |dA| =", float(np.abs(np.load(p)["A"] - np.load(q)["A"]).max()))
