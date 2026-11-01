"""TrackEval wrapper: evaluates every tracker folder and writes work/results.csv
(one row per tracker x sequence, plus TrackEval's COMBINED row)."""
import os, sys, glob, csv
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "TrackEval"))
import trackeval  # noqa: E402
import run         # noqa: E402

KEYS = [("HOTA", "HOTA"), ("HOTA", "DetA"), ("HOTA", "AssA"), ("Identity", "IDF1"),
        ("CLEAR", "IDSW"), ("CLEAR", "MOTA"), ("CLEAR", "Frag")]

def link_real_gt():
    d = os.path.join(run.WORK, "gt", "MOT17-train"); os.makedirs(d, exist_ok=True)
    for s in run.STATIC + run.MOVING:
        dst = os.path.join(d, s)
        if not os.path.exists(dst):
            os.symlink(os.path.join(run.SEQ_ROOT, s), dst)

def evaluate(bench, seqs=None, tag=None):
    """Evaluate trackers on `seqs` (default: all). TrackEval's COMBINED_SEQ row is renamed
    COMBINED_<tag> so per-group aggregates (static, moving, one synthetic level) can be kept."""
    tdir = os.path.join(run.WORK, "trackers", f"{bench}-train")
    trackers = sorted(os.listdir(tdir))
    seqs = seqs or sorted({os.path.basename(p)[:-4] for p in glob.glob(os.path.join(tdir, "*", "data", "*.txt"))})
    trackers = [t for t in trackers if all(os.path.exists(os.path.join(tdir, t, "data", s + ".txt")) for s in seqs)]
    gtdir = os.path.join(run.WORK, "gt", f"{bench}-train")
    lengths = {s: int(open(os.path.join(gtdir, s, "seqinfo.ini")).read().split("seqLength=")[1].split()[0]) for s in seqs}
    ev = trackeval.Evaluator(dict(trackeval.Evaluator.get_default_eval_config(),
                                  USE_PARALLEL=False, PRINT_RESULTS=False, PRINT_CONFIG=False,
                                  OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                                  TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True))
    cfg = dict(trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
               GT_FOLDER=os.path.join(run.WORK, "gt"), TRACKERS_FOLDER=os.path.join(run.WORK, "trackers"),
               BENCHMARK=bench, SPLIT_TO_EVAL="train", TRACKERS_TO_EVAL=trackers,
               SEQ_INFO=lengths, PRINT_CONFIG=False)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False)),
               trackeval.metrics.Identity(dict(PRINT_CONFIG=False))]
    res, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(cfg)], metrics)
    rows = []
    for tr, per in res["MotChallenge2DBox"].items():
        for seq, r in per.items():
            p = r["pedestrian"]
            if seq == "COMBINED_SEQ":
                if tag is None: continue
                seq = "COMBINED_" + tag
            elif tag is not None:
                continue
            row = dict(bench=bench, tracker=tr, seq=seq)
            for fam, k in KEYS:
                v = p[fam][k]
                row[k] = float(np.mean(v)) * (100 if k not in ("IDSW", "Frag") else 1)
            rows.append(row)
    return rows

if __name__ == "__main__":
    link_real_gt()
    import synth
    rows = evaluate("MOT17")
    have = lambda s: os.path.exists(os.path.join(run.WORK, "gt", "MOT17-train", s))
    for tag, group in (("static", run.STATIC), ("moving", run.MOVING), ("all", run.STATIC + run.MOVING)):
        if all(have(s) for s in group): rows += evaluate("MOT17", group, tag)
    if os.path.isdir(os.path.join(run.WORK, "trackers", "MOT17SYN-train")):
        rows += evaluate("MOT17SYN")
        for k, m in synth.LEVELS:
            group = [f"{s}_{k}{m}" for s in run.STATIC]
            rows += evaluate("MOT17SYN", group, f"{k}{m}")
    if os.path.isdir(os.path.join(run.WORK, "trackers", "MOT17STR-train")):
        rows += evaluate("MOT17STR")
        for s in run.STRIDES:
            for tag, group in (("static", run.STATIC), ("moving", run.MOVING)):
                rows += evaluate("MOT17STR", [f"{q}_stride{s}" for q in group], f"{tag}_stride{s}")
    with open(os.path.join(run.WORK, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "rows ->", os.path.join(run.WORK, "results.csv"))
