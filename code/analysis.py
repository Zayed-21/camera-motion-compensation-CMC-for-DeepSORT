"""Builds the paper's tables and figure from work/results.csv, work/cmc/*.npz, work/timing.json.
Outputs go to work/paper/: table_real.tex, table_synth.tex, table_time.tex, fig_dose.pdf,
summary.json (every number quoted in the text)."""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run, synth

W = run.WORK; OUT = os.path.join(W, "paper"); os.makedirs(OUT, exist_ok=True)
M = ["HOTA", "AssA", "IDF1", "IDSW", "MOTA"]
df = pd.read_csv(os.path.join(W, "results.csv"))
cmc_names = [f"cmc_s{s}" for s in run.SEEDS]

def cmc_stats(sub):
    c = sub[sub.tracker.isin(cmc_names)]
    return c.groupby("seq")[M + ["DetA"]].agg(["mean", "min", "max"])

def motion_severity(spec):
    info = run.frames(spec)[0].info
    z = np.load(os.path.join(W, "cmc", f"{spec}_s0.npz"))
    A, inf = z["A"][1:], z["info"][1:]
    ok = inf[:, 2] == 0
    t = np.hypot(A[ok, 0, 2], A[ok, 1, 2])
    return dict(sev=float(100 * np.median(t) / info["w"]) if ok.any() else float("nan"),
                fallback=float(100 * inf[:, 2].mean()), inlier=float(np.median(inf[:, 1])))

summary = {}
# ---------------- real sequences ----------------
real = df[df.bench == "MOT17"]
base = real[real.tracker == "base"].set_index("seq")
cs = cmc_stats(real)
rows = []
order = run.STATIC + ["COMBINED_static"] + run.MOVING + ["COMBINED_moving", "COMBINED_all"]
for s in order:
    if s not in base.index: continue
    r = dict(seq=s)
    if not s.startswith("COMBINED"):
        r.update(motion_severity(s))
    for k in M:
        r[k + "_b"] = base.loc[s, k]; r[k + "_c"] = cs.loc[s, (k, "mean")]
        r[k + "_lo"] = cs.loc[s, (k, "min")]; r[k + "_hi"] = cs.loc[s, (k, "max")]
        r[k + "_d"] = r[k + "_c"] - r[k + "_b"]
    rows.append(r)
R = pd.DataFrame(rows).set_index("seq"); summary["real"] = R.round(3).reset_index().to_dict("records")

def fmt(v, k):
    return f"{v:.0f}" if k == "IDSW" else f"{v:.1f}"

def sgn(v, k):
    if k == "IDSW":
        return "0" if abs(v) < 0.5 else f"{v:+.0f}".replace("-", "$-$")
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}".replace("-", "$-$")

lines = [r"\begin{tabular}{@{}lr" + "rr" * 4 + "@{}}", r"\toprule",
         r" & Sev. & \multicolumn{2}{c}{HOTA} & \multicolumn{2}{c}{AssA} & \multicolumn{2}{c}{IDF1} & \multicolumn{2}{c}{IDSW}\\",
         r"Sequence & (\%) & Base & $\Delta$ & Base & $\Delta$ & Base & $\Delta$ & Base & $\Delta$\\", r"\midrule"]
for s, r in R.iterrows():
    name = {"COMBINED_static": r"\textit{Static (comb.)}", "COMBINED_moving": r"\textit{Moving (comb.)}",
            "COMBINED_all": r"\textit{All (comb.)}"}.get(s, s.replace("MOT17-", ""))
    sev = "" if s.startswith("COMBINED") else f"{r.sev:.2f}"
    cells = " & ".join(f"{fmt(r[k + '_b'], k)} & {sgn(r[k + '_d'], k)}" for k in ["HOTA", "AssA", "IDF1", "IDSW"])
    lines.append(f"{name} & {sev} & {cells}\\\\")
    if s in ("COMBINED_static", "COMBINED_moving"): lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "table_real.tex"), "w").write("\n".join(lines))

# ---------------- synthetic dose-response ----------------
syn = df[df.bench == "MOT17SYN"]
lv = []
for k, m in synth.LEVELS:
    tag = f"COMBINED_{k}{m}"
    sub = syn[syn.seq == tag]
    if sub.empty: continue
    b = sub[sub.tracker == "base"].iloc[0]; o = sub[sub.tracker == "oracle"].iloc[0]
    c = sub[sub.tracker.isin(cmc_names)]
    fb = np.mean([np.load(p)["info"][1:, 2].mean() for s in run.STATIC
                  for p in glob.glob(os.path.join(W, "cmc", f"{s}_{k}{m}_s*.npz"))])
    d = dict(kind=k, mag=m, fallback=100 * fb)
    for key in ["HOTA", "AssA", "IDF1", "IDSW", "DetA"]:
        d[key + "_b"] = b[key]; d[key + "_o"] = o[key]
        d[key + "_c"] = c[key].mean(); d[key + "_clo"] = c[key].min(); d[key + "_chi"] = c[key].max()
    lv.append(d)
L = pd.DataFrame(lv); summary["synthetic"] = L.round(3).to_dict("records")

lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{IDF1} & \multicolumn{3}{c}{IDSW}\\",
         r"Motion & px & Base & CMC & Oracle & Base & CMC & Oracle\\", r"\midrule"]
for _, d in L.iterrows():
    lines.append(f"{d.kind} & {d.mag} & {d.IDF1_b:.1f} & {d.IDF1_c:.1f} & {d.IDF1_o:.1f} & "
                 f"{d.IDSW_b:.0f} & {d.IDSW_c:.0f} & {d.IDSW_o:.0f}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "table_synth.tex"), "w").write("\n".join(lines))

if len(L):
    # Paired effect vs. measured motion: dIDF1 = CMC (mean of seeds) - baseline, same detections.
    # x = median inter-frame camera displacement as % of image width (synthetic: true offsets;
    # real: estimated translation, seed 0). Baseline-vs-level comparisons are NOT plotted because
    # each level shows different visible content and detections.
    def syn_sev(k, m):
        d = []
        for s in run.STATIC:
            info = run.frames(f"{s}_{k}{m}")[0]
            o = info.offsets
            d.append(np.hypot(*np.diff(o, axis=0).T))
        return 100 * np.median(np.concatenate(d)) / info.crop[0]
    L["sev"] = [syn_sev(k, m) for k, m in zip(L.kind, L.mag)]
    summary["synthetic_sev"] = L[["kind", "mag", "sev"]].round(3).to_dict("records")
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    still = L[L.kind == "still"]
    for kind, col, mk, lab in [("pan", "#2a78d6", "o", "Synthetic pan"), ("shake", "#eb6834", "s", "Synthetic shake")]:
        sub = pd.concat([still, L[L.kind == kind]])
        ax.plot(sub.sev, sub.IDF1_c - sub.IDF1_b, marker=mk, color=col, lw=1.6, ms=4.5, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
    rs = R[~R.index.str.startswith("COMBINED")]
    st = rs.loc[[s for s in run.STATIC if s in rs.index]]; mv = rs.loc[[s for s in run.MOVING if s in rs.index]]
    ax.scatter(st.sev, st.IDF1_d, marker="o", facecolors="none", edgecolors="0.2", s=22, lw=0.9, label="Real, static camera", zorder=4)
    ax.scatter(mv.sev, mv.IDF1_d, marker="D", color="0.2", s=20, label="Real, moving camera", zorder=4)
    for s, r in mv.iterrows():
        off = {"MOT17-10": (-12, -10), "MOT17-11": (3, 5)}.get(s, (4, 4))
        ax.annotate(s.replace("MOT17-", ""), (r.sev, r.IDF1_d), textcoords="offset points", xytext=off, fontsize=6.5, color="0.15")
    ax.set_xlabel("Median camera motion (% of image width per frame)")
    ax.set_ylabel(r"$\Delta$IDF1, CMC $-$ baseline")
    ax.grid(alpha=.25, lw=.5)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_dose.pdf")); fig.savefig(os.path.join(OUT, "fig_dose.png"), dpi=220)

# ---------------- timing ----------------
tp = os.path.join(W, "timing.json")
if os.path.exists(tp):
    T = json.load(open(tp)); summary["timing"] = T
    u, e = T["update_ms"], T["cmc_estimate_ms"]
    q = lambda d: f"{d['median']:.2f} [{d['q1']:.2f}, {d['q3']:.2f}]"
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Component & ms/frame, median [IQR]\\", r"\midrule",
             f"DeepSORT update, baseline & {q(u['base'])}\\\\", f"DeepSORT update + state warp & {q(u['cmc'])}\\\\"]
    for k, v in sorted(e.items()):
        lines.append(f"CMC estimation, {k.replace('px', '')} px wide & {q(v)}\\\\")
    for k, v in sorted(T["detector_ms_median_by_seq"].items()):
        lines.append(f"YOLOv8n detection, {k.replace('px', '')} px wide$^\\dagger$ & {np.median(v):.1f}\\\\")
    for k, v in sorted(T["encoder_ms_median_by_seq"].items()):
        lines.append(f"Appearance encoding, {k.replace('px', '')} px wide$^\\dagger$ & {np.median(v):.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "table_time.tex"), "w").write("\n".join(lines))

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(R[[c for c in R.columns if c.endswith(("_b", "_d")) or c in ("sev", "fallback")]].round(2).to_string())
print(L.round(2).to_string() if len(L) else "no synthetic")
