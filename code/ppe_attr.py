"""PPE mis-attribution analysis (protocol item 4). For each tracker output and sequence:
match tracks to GT persons per frame (IoU > 0.5, Hungarian; GT pedestrians with flag 1);
a PPE observation of person p in frame t is credited to the matched track. Track owner = most-matched person.
id_err = share of credited observations whose person != owner;
viol_err = share whose fixed PPE status (non-compliant w.p. 0.2, seed 0, per person) != owner's status."""
import os, sys, csv, glob, numpy as np
from scipy.optimize import linear_sum_assignment

def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y2 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / (a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - i)

def credits(gt, tr):
    out = []
    by_t = {f: tr[tr[:, 0] == f] for f in np.unique(tr[:, 0])} if len(tr) else {}
    for f in np.unique(gt[:, 0]):
        g = gt[gt[:, 0] == f]; r = by_t.get(f)
        if r is None or not len(r): continue
        c = iou(g[:, 2:6], r[:, 2:6]); a, b = linear_sum_assignment(-c)
        out += [(int(g[i, 1]), int(r[j, 1])) for i, j in zip(a, b) if c[i, j] > 0.5]
    return out

def score(cr, status):
    if not cr: return np.nan, np.nan, 0
    from collections import Counter, defaultdict
    per = defaultdict(Counter)
    for p, t in cr: per[t][p] += 1
    owner = {t: c.most_common(1)[0][0] for t, c in per.items()}
    idw = sum(p != owner[t] for p, t in cr); vw = sum(status[p] != status[owner[t]] for p, t in cr)
    return 100 * idw / len(cr), 100 * vw / len(cr), len(cr)

def load(p, ncol=10):
    return np.loadtxt(p, delimiter=",", ndmin=2) if os.path.exists(p) and os.path.getsize(p) else np.zeros((0, ncol))

def run_group(gt_dirs, tdir, trackers, seqs, tag):
    rows = []
    for t in trackers:
        tot = []
        for s in seqs:
            gt = load(os.path.join(gt_dirs[s], "gt", "gt.txt"))
            gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)] if gt.shape[1] >= 8 else gt
            ids = np.unique(gt[:, 1]).astype(int); rng = np.random.default_rng([0, abs(hash(s)) % 2**31])
            rng = np.random.default_rng(sum(map(ord, s)))   # deterministic per sequence
            status = dict(zip(ids, rng.random(len(ids)) < 0.2))
            tr = load(os.path.join(tdir, t, "data", s + ".txt"))
            cr = credits(gt, tr); tot += [(s, p, q) for p, q in cr]
            i, v, n = score(cr, status); rows.append(dict(group=tag, tracker=t, seq=s, id_err=i, viol_err=v, n=n))
        # combined: pool credits, owners per (seq, track)
        from collections import Counter, defaultdict
        per = defaultdict(Counter)
        for s, p, q in tot: per[s, q][p] += 1
        own = {k: c.most_common(1)[0][0] for k, c in per.items()}
        sts = {}
        for s in seqs:
            gt = load(os.path.join(gt_dirs[s], "gt", "gt.txt")); gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
            ids = np.unique(gt[:, 1]).astype(int); rng = np.random.default_rng(sum(map(ord, s)))
            sts.update({(s, p): v for p, v in zip(ids, rng.random(len(ids)) < 0.2)})
        n = len(tot)
        rows.append(dict(group=tag, tracker=t, seq="COMBINED", n=n,
                         id_err=100 * sum(p != own[s, q] for s, p, q in tot) / max(n, 1),
                         viol_err=100 * sum(sts[s, p] != sts[s, own[s, q]] for s, p, q in tot) / max(n, 1)))
    return rows

def reports(gt_dirs, tdir, trackers, seqs, tag):
    """Added after the first results (exploratory): violation reports per non-compliant person.
    A report = one track whose owner is non-compliant; ideal = 1.0 (fragmentation inflates it)."""
    from collections import Counter, defaultdict
    rows = []
    for t in trackers:
        rep = per_n = 0
        for s in seqs:
            gt = load(os.path.join(gt_dirs[s], "gt", "gt.txt")); gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
            ids = np.unique(gt[:, 1]).astype(int); rng = np.random.default_rng(sum(map(ord, s)))
            st = dict(zip(ids, rng.random(len(ids)) < 0.2))
            cr = credits(gt, load(os.path.join(tdir, t, "data", s + ".txt")))
            per = defaultdict(Counter)
            for p, q in cr: per[q][p] += 1
            own = {q: c.most_common(1)[0][0] for q, c in per.items()}
            rep += sum(st[o] for o in own.values()); per_n += sum(st[p] for p in {p for p, _ in cr})
        rows.append(dict(group=tag, tracker=t, reports=rep, violators=per_n, per_violator=rep / max(per_n, 1)))
    return rows
