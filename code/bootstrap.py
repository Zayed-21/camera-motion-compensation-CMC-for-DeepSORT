"""C1: paired circular moving-block bootstrap of tracking-score differences (Kunsch, 1989).
For resample r, each sequence's frames are replaced by ceil(n/B) blocks of B consecutive
frames with random start (wrapping at the end); the SAME frame list is applied to the ground
truth and to every tracker, frames are renumbered 1..n, and TrackEval recomputes HOTA and IDF1
for each sequence and for the TrackEval-combined group. Identities are kept, so a person
visible in two drawn blocks is one identity; block joins add a small, equal penalty to every
condition. Output: work/bootstrap.csv (one row per resample x group x tracker)."""
import os, sys, csv, time, shutil
import numpy as np
import run, evaluate, trackeval
if os.environ.get("SEQS"): run.STATIC, run.MOVING = [], os.environ["SEQS"].split(",")

B, SEED = 60, 2026
TRACKERS = os.environ.get("BTRACKERS", "base,cmc_s0,trans,ecc,hybrid").split(",")
WK, NW = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))  # parallel shards: r % NW == WK
GROUP = os.environ.get("GROUP", "full")      # "full" or "stride2" / "stride3" / "stride6" (blocks of processed frames)
OUT = os.path.join(run.WORK, "boot", f"{GROUP}_w{WK}.csv"); os.makedirs(os.path.dirname(OUT), exist_ok=True)
TMP = os.path.join(run.WORK, f"boot_tmp{WK}")
if GROUP == "full":
    SEQS, TBENCH = run.MOVING, "MOT17-train"
    GT = lambda s: os.path.join(run.SEQ_ROOT, s)
elif GROUP.startswith("syn_"):                # synthetic / virtual-camera level over the static sequences
    SEQS, TBENCH = [f"{q}_{GROUP[4:]}" for q in (run.STATIC or run.MOVING)], "MOT17SYN-train"
    GT = lambda s: os.path.join(run.WORK, "gt", "MOT17SYN-train", s)
else:
    SEQS, TBENCH = [f"{q}_{GROUP}" for q in run.MOVING], "MOT17STR-train"
    GT = lambda s: os.path.join(run.WORK, "gt", "MOT17STR-train", s)

def frames_for(n, rng):
    starts = rng.integers(0, n, size=-(-n // B))
    return (np.concatenate([(s + np.arange(B)) % n for s in starts])[:n] + 1).astype(int)

def remap(rows, frames):
    by = {int(f): rows[rows[:, 0] == f] for f in np.unique(rows[:, 0])}
    out = []
    for k, f in enumerate(frames, 1):
        r = by.get(int(f))
        if r is not None and len(r):
            r = r.copy(); r[:, 0] = k; out.append(r)
    return np.concatenate(out) if out else rows[:0]

def load(p):
    return np.loadtxt(p, delimiter=",", ndmin=2) if os.path.getsize(p) else np.zeros((0, 10))

def one(r, seqs):
    rng = np.random.default_rng([SEED, r])
    bench = f"BOOT{r}"
    gt_root = os.path.join(TMP, "gt"); tr_root = os.path.join(TMP, "trackers")
    shutil.rmtree(TMP, ignore_errors=True)
    for s in seqs:
        info = run.common.seqinfo(GT(s))
        fr = frames_for(info["n"], rng)
        gt = load(os.path.join(GT(s), "gt", "gt.txt"))
        d = os.path.join(gt_root, f"{bench}-train", s, "gt"); os.makedirs(d)
        np.savetxt(os.path.join(d, "gt.txt"), remap(gt, fr), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.6f")
        open(os.path.join(gt_root, f"{bench}-train", s, "seqinfo.ini"), "w").write(
            f"[Sequence]\nname={s}\nimDir=img1\nframeRate={info['fps']:g}\nseqLength={info['n']}\n"
            f"imWidth={info['w']}\nimHeight={info['h']}\nimExt=.jpg\n")
        for t in TRACKERS:
            res = load(os.path.join(run.WORK, "trackers", TBENCH, t, "data", s + ".txt"))
            d = os.path.join(tr_root, f"{bench}-train", t, "data"); os.makedirs(d, exist_ok=True)
            np.savetxt(os.path.join(d, s + ".txt"), remap(res, fr), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d")
    ev = trackeval.Evaluator(dict(trackeval.Evaluator.get_default_eval_config(), USE_PARALLEL=False,
                                  PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False,
                                  OUTPUT_DETAILED=False, PLOT_CURVES=False, TIME_PROGRESS=False,
                                  DISPLAY_LESS_PROGRESS=True))
    cfg = dict(trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(), GT_FOLDER=gt_root,
               TRACKERS_FOLDER=tr_root, BENCHMARK=bench, SPLIT_TO_EVAL="train", TRACKERS_TO_EVAL=TRACKERS,
               SEQ_INFO={s: run.common.seqinfo(GT(s))["n"] for s in seqs},
               PRINT_CONFIG=False)
    res, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(cfg)],
                         [trackeval.metrics.HOTA(), trackeval.metrics.Identity(dict(PRINT_CONFIG=False))])
    rows = []
    for t, per in res["MotChallenge2DBox"].items():
        for s, v in per.items():
            p = v["pedestrian"]
            rows.append(dict(r=r, tracker=t, seq=s, HOTA=100 * float(np.mean(p["HOTA"]["HOTA"])),
                             IDF1=100 * float(p["Identity"]["IDF1"])))
    return rows

if __name__ == "__main__":
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    budget = float(os.environ.get("BUDGET", 150))
    done = set()
    if os.path.exists(OUT):
        done = {int(x["r"]) for x in csv.DictReader(open(OUT))}
    t0 = time.time(); new = False
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["r", "tracker", "seq", "HOTA", "IDF1"])
        if not done: w.writeheader()
        for r in range(WK, n_target, NW):
            if r in done: continue
            if time.time() - t0 > budget: break
            w.writerows(one(r, SEQS)); f.flush()
    n = len({int(x["r"]) for x in csv.DictReader(open(OUT))})
    print("resamples done:", n, "elapsed", round(time.time() - t0))
