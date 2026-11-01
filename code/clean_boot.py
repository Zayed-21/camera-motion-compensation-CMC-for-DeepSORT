"""Drop incomplete resamples (a command killed mid-write) so bootstrap.py redoes them."""
import csv, glob, collections, sys
for f in glob.glob("work*/boot/*.csv"):
    try: rows = [r for r in csv.DictReader(open(f)) if r.get("IDF1") not in (None, "")]
    except Exception as e: print("skip", f, e); continue
    n = collections.Counter(r["r"] for r in rows); full = max(n.values()) if n else 0
    keep = [r for r in rows if n[r["r"]] == full]
    if len(keep) != len(rows) or True:
        w = csv.DictWriter(open(f, "w", newline=""), fieldnames=["r", "tracker", "seq", "HOTA", "IDF1"]); w.writeheader(); w.writerows(keep)
    print(f, len(set(r["r"] for r in keep)), "complete; dropped", len(rows) - len(keep))
