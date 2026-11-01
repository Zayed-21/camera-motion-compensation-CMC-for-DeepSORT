import os, csv, run, evaluate
run.STATIC, run.MOVING = [], ["ETH-Pedcross2"]
run.detect("ETH-Pedcross2")
for m in ("base", "cmc", "ecc"): run.track("ETH-Pedcross2", m, 0)
evaluate.link_real_gt()
for r in evaluate.evaluate("MOT17", ["ETH-Pedcross2"]): print(r["tracker"], round(r["HOTA"], 2), round(r["IDF1"], 2), r["IDSW"])
