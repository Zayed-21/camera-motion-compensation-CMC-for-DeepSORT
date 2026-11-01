"""Identity-merge check: for each predicted track, how many distinct GT pedestrians it is matched to
(per-frame Hungarian matching at IoU >= 0.5, pedestrians class 1 only). A track matched to >1 GT
identity has been continued onto another person. Writes work/paper/merges.csv."""
import os, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
import run, synth

def iou(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter)

def merges(gt_file, res_file):
    gt = np.loadtxt(gt_file, delimiter=","); gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    tr = np.loadtxt(res_file, delimiter=",", ndmin=2)
    seen = {}
    for f in np.unique(tr[:, 0]):
        g, t = gt[gt[:, 0] == f], tr[tr[:, 0] == f]
        if not len(g): continue
        m = iou(t[:, 2:6], g[:, 2:6])
        r, c = linear_sum_assignment(-m)
        for i, j in zip(r, c):
            if m[i, j] >= 0.5:
                seen.setdefault(int(t[i, 1]), set()).add(int(g[j, 1]))
    n = np.array([len(v) for v in seen.values()])
    return dict(tracks=len(n), merged=int((n > 1).sum()), pct_merged=100 * (n > 1).mean())

rows = []
for s in run.STATIC + run.MOVING:
    for tr in ("base", "cmc_s0"):
        rows.append(dict(seq=s, tracker=tr, **merges(os.path.join(run.SEQ_ROOT, s, "gt", "gt.txt"),
                    os.path.join(run.WORK, "trackers", "MOT17-train", tr, "data", s + ".txt"))))
for s in run.STATIC:
    for k, m in synth.LEVELS:
        sp = f"{s}_{k}{m}"
        for tr in ("base", "cmc_s0"):
            rows.append(dict(seq=sp, tracker=tr, **merges(os.path.join(run.WORK, "gt", "MOT17SYN-train", sp, "gt", "gt.txt"),
                        os.path.join(run.WORK, "trackers", "MOT17SYN-train", tr, "data", sp + ".txt"))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(run.WORK, "paper", "merges.csv"), index=False)
print(df.pivot_table(index="seq", columns="tracker", values=["tracks", "merged", "pct_merged"]).round(1).to_string())
