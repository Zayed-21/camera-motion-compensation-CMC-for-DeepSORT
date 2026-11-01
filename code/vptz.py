"""Virtual PTZ / pole-sway camera from static footage (protocol: SURVEILLANCE_PROTOCOL.md).
A rotating camera at the same optical centre sees H = K R K^-1 applied to the reference image (exact for
any depth). Output: centred CW x CH window. kinds: 'pstill' (0), 'ptz' (deg/s), 'sway' (amplitude, 1/100 deg)."""
import os, numpy as np, cv2
F0 = 1662.8   # focal length for 1920-px width (60 deg HFOV); scaled with width
def geom(w, h):
    """Focal length and output window (2/3 of the frame) for a w x h reference frame; 1920x1080 -> 1280x720."""
    return F0 * w / 1920, int(round(w * 2 / 3)), int(round(h * 2 / 3))
F, CW, CH = 1662.8, 1280, 720   # MOT17 values (kept for reference)
KINDS = ("pstill", "ptz", "sway")

def angles(kind, mag, n, fps, seed=0):
    t = np.arange(n) / fps; pan = np.zeros(n); tilt = np.zeros(n)
    if kind == "ptz":
        th, w, dwell = 8.0, float(mag), 1.0
        seg = [(0.0, th, th / w), (th, th, dwell), (th, -th, 2 * th / w), (-th, -th, dwell), (-th, 0.0, th / w)]
        T = sum(s[2] for s in seg)
        tt = t % T; out = np.zeros(n); t0 = 0.0
        for a, b, d in seg:
            m = (tt >= t0) & (tt < t0 + d); out[m] = a + (b - a) * (tt[m] - t0) / d; t0 += d
        pan = out
    elif kind == "sway":
        a = mag / 100.0
        pan = a * np.sin(2 * np.pi * 0.8 * t); tilt = 0.6 * a * np.sin(2 * np.pi * 1.1 * t)
        rng = np.random.default_rng(seed); tb = 0.0
        while True:
            tb += rng.exponential(1 / 0.25)
            if tb >= t[-1]: break
            m = t >= tb; d = t[m] - tb
            pan[m] += 2 * a * np.exp(-d / 0.5) * np.sin(2 * np.pi * 3.0 * d)
    return np.deg2rad(pan), np.deg2rad(tilt)

def K(w, h): f = geom(w, h)[0]; return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])

def hom(pan, tilt, w, h):
    """Reference image -> output window."""
    cp, sp, ct, st = np.cos(pan), np.sin(pan), np.cos(tilt), np.sin(tilt)
    Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]]); Rx = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])
    k = K(w, h); _, cw, ch = geom(w, h); T = np.array([[1, 0, -(w - cw) / 2], [0, 1, -(h - ch) / 2], [0, 0, 1.0]])
    return T @ k @ Rx @ Ry @ np.linalg.inv(k)

def sim_fit(G, CW=1280, CH=720):
    """Least-squares similarity [M|t] approximating homography G on a 9x9 grid over the window."""
    xs, ys = np.meshgrid(np.linspace(0, CW, 9), np.linspace(0, CH, 9)); p = np.c_[xs.ravel(), ys.ravel()]
    q = cv2.perspectiveTransform(p[None].astype(np.float64), G)[0]
    A, _ = cv2.estimateAffinePartial2D(p.astype(np.float32), q.astype(np.float32), method=cv2.LMEDS)
    return A.astype(np.float64)

class Frames:
    def __init__(self, seq_dir, kind, mag, seqinfo):
        self.dir, self.info = seq_dir, seqinfo(seq_dir); n, fps = self.info["n"], self.info["fps"]
        self.pan, self.tilt = angles(kind, mag, n, fps); self.n = n
        self.H = [hom(p, q, self.info["w"], self.info["h"]) for p, q in zip(self.pan, self.tilt)]
        _, self.cw, self.ch = geom(self.info["w"], self.info["h"])
    def __call__(self, i):
        img = cv2.imread(os.path.join(self.dir, "img1", f"{i:06d}{self.info['ext']}"))
        return cv2.warpPerspective(img, self.H[i - 1], (self.cw, self.ch), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    def oracle(self):
        A = [np.eye(2, 3)] + [sim_fit(self.H[i] @ np.linalg.inv(self.H[i - 1]), self.cw, self.ch) for i in range(1, self.n)]
        return np.stack(A)
    def telemetry(self, sigma_deg, lat, seed=0):
        rng = np.random.default_rng(seed); s = np.deg2rad(sigma_deg)
        idx = np.maximum(np.arange(self.n) - lat, 0)
        p = self.pan[idx] + rng.normal(0, s, self.n); q = self.tilt[idx] + rng.normal(0, s, self.n)
        Hr = [hom(a, b, self.info["w"], self.info["h"]) for a, b in zip(p, q)]
        A = [np.eye(2, 3)] + [sim_fit(Hr[i] @ np.linalg.inv(Hr[i - 1]), self.cw, self.ch) for i in range(1, self.n)]
        return np.stack(A)
    def gt(self, gt, min_vis=0.25):
        out = []
        for r in gt:
            Hm = self.H[int(r[0]) - 1]; x, y, w, h = r[2:6]
            c = cv2.perspectiveTransform(np.array([[[x, y], [x + w, y], [x, y + h], [x + w, y + h]]], np.float64), Hm)[0]
            x1, y1 = c.min(0); x2, y2 = c.max(0)
            iw = max(0, min(x2, self.cw) - max(x1, 0)); ih = max(0, min(y2, self.ch) - max(y1, 0))
            frac = iw * ih / max((x2 - x1) * (y2 - y1), 1e-9)
            if frac <= 0: continue
            o = r.copy(); o[2:6] = x1, y1, x2 - x1, y2 - y1; o[8] = o[8] * frac
            if frac < min_vis and o[7] == 1: o[7] = 8
            out.append(o)
        return np.array(out)

if __name__ == "__main__":
    p, _ = angles("ptz", 10, 900, 30); assert abs(np.rad2deg(p).max() - 8) < 0.1 and abs(np.rad2deg(p).min() + 8) < 0.1
    G = hom(np.deg2rad(0.5), 0, 1920, 1080) @ np.linalg.inv(hom(0, 0, 1920, 1080)); A = sim_fit(G)
    assert abs(A[0, 2] + F * np.tan(np.deg2rad(0.5))) < 1.5, A
    assert geom(1920, 1080) == (F, CW, CH) and geom(768, 576) == (F * 0.4, 512, 384)   # 0.5 deg pan right shifts scene left ~14.5 px
    print("vptz self-check ok", A.round(3).tolist())
