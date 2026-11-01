"""Uncertainty of the PPE measures over which people are non-compliant: 1,000 redraws of the
per-person status (p = 0.2), paired between baseline and CMC. 95% percentile intervals of the
difference (CMC - base) in violation reports per non-compliant person and in violation error (%).
Stage 1 (time-bounded, resumable) caches matches in ppe_cache/; stage 2 does the redraws."""
import os, sys, json, time, pickle, numpy as np, ppe_attr as P
from collections import Counter, defaultdict
exec(open("ppe_run.py").read().split("rows = []")[0])
os.makedirs("ppe_cache", exist_ok=True); t0 = time.time()
for gi, (tag, gd, td, tr, sq) in enumerate(G):
    f = f"ppe_cache/{gi}.pkl"
    if os.path.exists(f): continue
    if time.time() - t0 > 120: print("pause"); sys.exit()
    cr = {}
    for t in ("base", "cmc_s0"):
        for s in sq:
            gt = P.load(gd[s] + "/gt/gt.txt"); gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
            cr[t, s] = np.array(P.credits(gt, P.load(f"{td}/{t}/data/{s}.txt")), int).reshape(-1, 2)
    pickle.dump((tag, sq, cr), open(f, "wb")); print("cached", tag, round(time.time() - t0), flush=True)
out = {}
for gi in range(len(G)):
    tag, sq, cr = pickle.load(open(f"ppe_cache/{gi}.pkl", "rb"))
    rng = np.random.default_rng(2026); R = 1000
    pre = {}
    for t in ("base", "cmc_s0"):
        for s in sq:
            c = cr[t, s]; per = defaultdict(Counter)
            for p, q in c: per[q][p] += 1
            own = {q: cc.most_common(1)[0][0] for q, cc in per.items()}
            pre[t, s] = (c[:, 0], np.array([own[q] for q in c[:, 1]]), np.array(list(own.values())), np.unique(c[:, 0]))
    maxid = {s: int(max(cr[t, s][:, 0].max() if len(cr[t, s]) else 0 for t in ("base", "cmc_s0"))) + 1 for s in sq}
    st = {s: rng.random((R, maxid[s])) < 0.2 for s in sq}
    res = {}
    for t in ("base", "cmc_s0"):
        rep = sum(st[s][:, pre[t, s][2]].sum(1) for s in sq); nv = sum(st[s][:, pre[t, s][3]].sum(1) for s in sq)
        n = sum(len(pre[t, s][0]) for s in sq)
        v = sum((st[s][:, pre[t, s][0]] != st[s][:, pre[t, s][1]]).sum(1) for s in sq)
        res[t] = (rep / np.maximum(nv, 1), 100 * v / max(n, 1))
    dr = res["cmc_s0"][0] - res["base"][0]; dv = res["cmc_s0"][1] - res["base"][1]
    out[tag] = dict(reports=[round(float(np.percentile(dr, q)), 2) for q in (2.5, 97.5)],
                    viol=[round(float(np.percentile(dv, q)), 2) for q in (2.5, 97.5)])
    print(tag, out[tag], flush=True)
json.dump(out, open("ppe_ci.json", "w"), indent=1)
