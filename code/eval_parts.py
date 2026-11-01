"""evaluate.py split into parts that each fit a short command; writes work/res_parts/<part>.csv,
then `python eval_parts.py merge` writes work/results.csv (same layout as evaluate.py)."""
import os, sys, csv, glob, time
import evaluate, run, synth
D = run.path("res_parts", "x").rsplit("/", 1)[0]

def parts():
    syn = os.path.isdir(os.path.join(run.WORK, "trackers", "MOT17SYN-train"))   # FRCNN run has no synthetic part
    p = ["real"] + (["syn_all"] + [f"syn_{k}{m}" for k, m in synth.LEVELS] if syn else []) + ["str_all"]
    p += [f"str_{t}_{s}" for s in run.STRIDES for t in ("static", "moving")]
    return p

def do(part):
    if part == "real":
        evaluate.link_real_gt(); rows = evaluate.evaluate("MOT17")
        for tag, g in (("static", run.STATIC), ("moving", run.MOVING), ("all", run.STATIC + run.MOVING)):
            rows += evaluate.evaluate("MOT17", g, tag)
    elif part == "syn_all":
        rows = evaluate.evaluate("MOT17SYN")
    elif part.startswith("syn_"):
        lv = part[4:]; rows = evaluate.evaluate("MOT17SYN", [f"{s}_{lv}" for s in run.STATIC], lv)
    elif part == "str_all":
        rows = evaluate.evaluate("MOT17STR")
    else:
        _, t, s = part.split("_"); g = run.STATIC if t == "static" else run.MOVING
        rows = evaluate.evaluate("MOT17STR", [f"{q}_stride{s}" for q in g], f"{t}_stride{s}")
    with open(os.path.join(D, part + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    if sys.argv[1:] == ["merge"]:
        out = []
        for p in parts():
            out += list(csv.DictReader(open(os.path.join(D, p + ".csv"))))
        with open(os.path.join(run.WORK, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
        print(len(out), "rows"); sys.exit()
    t0 = time.time()
    for p in parts():
        if os.path.exists(os.path.join(D, p + ".csv")): continue
        if time.time() - t0 > 100: print("pause"); break
        do(p); print("evaluated", p, round(time.time() - t0), flush=True)
    else:
        print("ALL EVALUATED")
