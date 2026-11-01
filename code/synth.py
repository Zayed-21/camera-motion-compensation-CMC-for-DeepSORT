"""Synthetic camera motion: a fixed-size crop window (80% of width and height) moved per frame
over a static-camera sequence. Translation only, so ground truth shifts exactly.
GT boxes with < 25% of their area inside the window become distractors (class 8),
so a tracker is neither rewarded nor penalised for them; boxes fully outside are dropped."""
import numpy as np

CROP = 0.8
LEVELS = [("still", 0), ("pan", 2), ("pan", 5), ("pan", 10), ("pan", 20),
          ("shake", 2), ("shake", 5), ("shake", 10)]

def crop_size(w, h):
    return int(round(w * CROP)), int(round(h * CROP))

def offsets(kind, mag, n, w, h, seed=0):
    cw, ch = crop_size(w, h)
    rx, ry = w - cw, h - ch                       # free range
    cx, cy = rx // 2, ry // 2
    if kind == "still" or mag == 0:
        o = np.tile([cx, cy], (n, 1))
    elif kind == "pan":                            # horizontal back-and-forth, constant speed
        period = 2 * rx
        p = (np.arange(n) * mag + rx // 2) % period
        x = np.where(p <= rx, p, period - p)
        o = np.c_[x, np.full(n, cy)]
    elif kind == "shake":                          # i.i.d. Gaussian position jitter about centre
        rng = np.random.default_rng(seed)
        j = rng.normal(0, mag, (n, 2))
        o = np.c_[np.clip(cx + j[:, 0], 0, rx), np.clip(cy + j[:, 1], 0, ry)]
    else:
        raise ValueError(kind)
    return np.rint(o).astype(int), (cw, ch)

def shift_gt(gt, offs, crop, min_vis=0.25):
    """gt: MOT17 gt rows (frame,id,x,y,w,h,flag,class,vis). Returns shifted rows."""
    cw, ch = crop
    o = offs[gt[:, 0].astype(int) - 1]
    x1, y1 = gt[:, 2] - o[:, 0], gt[:, 3] - o[:, 1]
    x2, y2 = x1 + gt[:, 4], y1 + gt[:, 5]
    iw = np.clip(np.minimum(x2, cw) - np.maximum(x1, 0), 0, None)
    ih = np.clip(np.minimum(y2, ch) - np.maximum(y1, 0), 0, None)
    frac = iw * ih / np.maximum(gt[:, 4] * gt[:, 5], 1e-9)
    out = gt.copy(); out[:, 2], out[:, 3] = x1, y1
    out[:, 8] = out[:, 8] * frac                   # visibility scaled by in-window fraction
    out[(frac < min_vis) & (out[:, 7] == 1), 7] = 8   # pedestrians mostly outside -> distractor
    return out[frac > 0]

if __name__ == "__main__":
    o, c = offsets("pan", 20, 100, 1920, 1080)
    assert c == (1536, 864) and o[:, 0].min() >= 0 and o[:, 0].max() <= 384
    assert set(np.abs(np.diff(o[:, 0]))) <= {0, 20, 8, 12, 4, 16}  # reflections at the ends
    assert (o[:, 1] == 108).all()
    o2, _ = offsets("still", 0, 5, 1920, 1080); assert (o2 == [192, 108]).all()
    g = np.array([[1, 1, 200, 200, 50, 100, 1, 1, 1.0],     # inside
                  [1, 2, 150, 200, 50, 100, 1, 1, 1.0],     # 8/50 inside -> distractor
                  [1, 3, 0, 0, 50, 50, 1, 1, 1.0]], float)  # outside -> dropped
    s = shift_gt(g, np.array([[192, 108]]), c)
    assert len(s) == 2 and s[0, 2] == 8 and s[0, 7] == 1 and s[1, 7] == 8
    print("synth self-check passed")
