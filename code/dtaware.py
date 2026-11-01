"""Revision 4: time-aware Kalman sensitivity for frame skipping.
Default ("step") convention: one KF step per processed frame, velocity in px per processed frame,
process noise per step, max_age counted in processed frames (30 -> 30k source frames).
Time-aware convention: F uses dt=k (velocity in px per source frame), process-noise variance
scaled by k (random-walk per source frame), max_age = ceil(30/k) processed frames (~30 source
frames), n_init and the appearance budget unchanged. Baseline and CMC share these settings.
Writes trackers to $WORK (a separate folder with symlinked dets/cmc/gt).
usage: WORK=... DET=... python dtaware.py <stride>"""
import sys, math, os
import numpy as np
import run
if os.environ.get("SEQS"): run.STATIC, run.MOVING = [], os.environ["SEQS"].split(",")
sys.path.insert(0, os.path.join(run.ROOT, "deep_sort"))
from deep_sort import kalman_filter as kfm

if os.environ.get("ALLDET"): run.DS["min_confidence"] = -1e9   # MOT15 ACF boxes used as provided
k = int(sys.argv[1]); VAR = os.environ.get("VAR", "full")   # full | kf (dt+Q only) | age (max_age only)
_init, _pred = kfm.KalmanFilter.__init__, kfm.KalmanFilter.predict
def init(self):
    _init(self)
    for i in range(4): self._motion_mat[i, 4 + i] = float(k)
def predict(self, mean, cov):
    m, c = _pred(self, mean, cov)            # adds Q once; add (k-1) more Q to scale variance by k
    std = np.r_[[self._std_weight_position * mean[3]] * 2, 1e-2, self._std_weight_position * mean[3],
                [self._std_weight_velocity * mean[3]] * 2, 1e-5, self._std_weight_velocity * mean[3]]
    return m, c + (k - 1) * np.diag(np.square(std))
if VAR in ("full", "kf"): kfm.KalmanFilter.__init__, kfm.KalmanFilter.predict = init, predict
if VAR in ("full", "age"): run.DS["max_age"] = math.ceil(30 / k)
for q in run.MOVING:
    sp = f"{q}_stride{k}"
    for mode in ("base", "cmc", "ecc", "hybrid"):
        out = run.path("trackers", "MOT17STR-train", "cmc_s0" if mode == "cmc" else mode, "data", sp + ".txt")
        if not os.path.exists(out): run.track(sp, mode, 0)
    print("done", sp, flush=True)
