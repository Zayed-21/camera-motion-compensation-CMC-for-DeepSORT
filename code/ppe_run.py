import os, csv, ppe_attr as P
H = os.path.expanduser("~/exp/code"); M17 = os.path.join(H, "data/MOT17/train")
MOV = ["MOT17-05", "MOT17-10", "MOT17-11", "MOT17-13"]; STA = ["MOT17-02", "MOT17-04", "MOT17-09"]
T = ["base", "cmc_s0", "ecc", "hybrid"]
G = []
G.append(("MOT17 moving k=1", {s: f"{M17}/{s}" for s in MOV}, f"{H}/work/trackers/MOT17-train", T, MOV))
sp = [f"{s}_stride6" for s in MOV]
G.append(("MOT17 moving k=6 per-step", {s: f"{H}/work/gt/MOT17STR-train/{s}" for s in sp}, f"{H}/work/trackers/MOT17STR-train", T, sp))
G.append(("MOT17 moving k=6 time-aware", {s: f"{H}/work/gt/MOT17STR-train/{s}" for s in sp}, f"{H}/work_dt/trackers/MOT17STR-train", T, sp))
m15 = ["ETH-Bahnhof", "ETH-Sunnyday", "KITTI-13"]
G.append(("MOT15 new moving k=1", {s: f"{H}/../mot15/{s}" for s in m15}, f"{H}/work_mot15/trackers/MOT17-train", T, m15))
for lv in ["pan20", "shake10"]:
    sq = [f"{s}_{lv}" for s in STA]
    G.append((f"synthetic {lv}", {s: f"{H}/work/gt/MOT17SYN-train/{s}" for s in sq}, f"{H}/work/trackers/MOT17SYN-train", ["base", "cmc_s0", "oracle"], sq))
for lv in ["pstill0", "ptz5", "ptz10", "ptz20", "ptz40", "sway10", "sway25", "sway50"]:
    sq = [f"{s}_{lv}" for s in STA]
    G.append((f"surveillance {lv}", {s: f"{H}/work_surv/gt/MOT17SYN-train/{s}" for s in sq}, f"{H}/work_surv/trackers/MOT17SYN-train", ["base", "cmc_s0", "ecc", "oracle"], sq))
rows = []; R2 = []
for tag, gd, td, tr, sq in G:
    rows += P.run_group(gd, td, tr, sq, tag); R2 += P.reports(gd, td, tr, sq, tag)
    print(tag, "REPORTS", " | ".join(f"{r['tracker']}: {r['per_violator']:.2f}" for r in R2 if r["group"] == tag), flush=True)
    c = [r for r in rows if r["group"] == tag and r["seq"] == "COMBINED"]
    print(tag, " | ".join(f"{r['tracker']}: id {r['id_err']:.2f}% viol {r['viol_err']:.2f}%" for r in c), flush=True)
with open(f"{H}/ppe_attr.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f"{H}/ppe_reports.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(R2[0])); w.writeheader(); w.writerows(R2)
