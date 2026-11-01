"""Runtime protocol: per-frame tracker-update time for base and CMC (transforms cached, so
only the warp is added), five repeats each on the real sequences; CMC estimation time comes
from the five seeded estimation runs; detector/encoder time from the detection stage.
Writes work/timing.json. Medians and interquartile ranges in ms/frame."""
import json, os, platform
import numpy as np, cv2
import run

def stats(x):
    x = np.asarray(x) * 1.0
    return dict(median=float(np.median(x)), q1=float(np.percentile(x, 25)), q3=float(np.percentile(x, 75)), n=int(x.size))

if __name__ == "__main__":
    seqs = run.STATIC + run.MOVING
    upd = {"base": [], "cmc": []}
    for s in seqs:
        for rep in range(5):
            upd["base"].append(run.track(s, "base", write=False) * 1e3)
            upd["cmc"].append(run.track(s, "cmc", rep, write=False) * 1e3)
    est = {}
    for s in seqs:
        w = run.frames(s)[0].info["w"]
        key = f"{w}px"
        est.setdefault(key, [])
        for seed in run.SEEDS:
            est[key].append(np.load(os.path.join(run.WORK, "cmc", f"{s}_s{seed}.npz"))["info"][1:, 3])
    det = {}
    for s in seqs:
        j = json.load(open(os.path.join(run.WORK, "dets", s + ".json")))
        det.setdefault(f"{run.frames(s)[0].info['w']}px", []).append(j)
    out = dict(
        update_ms={k: stats(np.concatenate(v)) for k, v in upd.items()},
        cmc_estimate_ms={k: stats(np.concatenate(v)) for k, v in est.items()},
        detector_ms_median_by_seq={k: [d["det_ms"] for d in v] for k, v in det.items()},
        encoder_ms_median_by_seq={k: [d["enc_ms"] for d in v] for k, v in det.items()},
        hardware=dict(cpu=open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip(),
                      cores=os.cpu_count(), ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
                      os=platform.platform(), python=platform.python_version(),
                      opencv=cv2.__version__, numpy=np.__version__))
    json.dump(out, open(os.path.join(run.WORK, "timing.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
