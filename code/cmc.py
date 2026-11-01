"""Camera-motion compensation for DeepSORT (our implementation).

estimate(): global similarity transform between frames t-1 and t from Shi-Tomasi corners
tracked with pyramidal Lucas-Kanade, fitted by RANSAC (cv2.estimateAffinePartial2D).
Person boxes of frame t-1 are masked out. Falls back to identity when tracking is poor.

warp_tracks(): applies A = [M | t] to each DeepSORT track's Kalman mean and covariance
(state u, v, a, h, du, dv, da, dh), following the form used by BoT-SORT, adapted to
DeepSORT's state.
"""
import cv2
import numpy as np

SCALE = 0.5          # estimate at half resolution
MAX_CORNERS = 1000
MIN_POINTS = 30
MIN_INLIER_RATIO = 0.5
RANSAC_PX = 3.0      # reprojection threshold in full-resolution pixels

def to_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)

def track_points(prev_gray, gray, prev_boxes):
    """Corners in t-1 (people masked) tracked into t. Returns (a, b) point arrays or None."""
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for x, y, w, h in np.asarray(prev_boxes, float).reshape(-1, 4) * SCALE:
        mask[max(int(y), 0):int(y + h) + 1, max(int(x), 0):int(x + w) + 1] = 0
    p0 = cv2.goodFeaturesToTrack(prev_gray, MAX_CORNERS, 0.01, 8, mask=mask, blockSize=3)
    if p0 is None or len(p0) < MIN_POINTS:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1) == 1
    return p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]

def fit(pts):
    """RANSAC similarity fit. Returns (A 2x3 full-res, n_tracked, inlier_ratio, fallback)."""
    ident = np.eye(2, 3)
    if pts is None or len(pts[0]) < MIN_POINTS:
        return ident, 0 if pts is None else len(pts[0]), 0.0, True
    a, b = pts
    A, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                         ransacReprojThreshold=RANSAC_PX * SCALE)
    ratio = float(inl.mean()) if inl is not None else 0.0
    if A is None or ratio < MIN_INLIER_RATIO:
        return ident, len(a), ratio, True
    A = A.copy(); A[:, 2] /= SCALE           # back to full-resolution pixels
    return A, len(a), ratio, False

def estimate(prev_gray, gray, prev_boxes):
    """Global similarity t-1 -> t: (A, n_tracked, inlier_ratio, fallback)."""
    return fit(track_points(prev_gray, gray, prev_boxes))

def oracle(offsets, i):
    """True transform for a translating crop: a static point moves by -(o_i - o_{i-1})."""
    A = np.eye(2, 3)
    A[:, 2] = -(offsets[i - 1] - offsets[i - 2]) if i >= 2 else 0
    return A

def warp_state(mean, cov, A):
    M, t = A[:, :2], A[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(M))))
    T = np.eye(8)
    T[0:2, 0:2] = M; T[3, 3] = s
    T[4:6, 4:6] = M; T[7, 7] = s
    mean = T @ mean; mean[0:2] += t
    return mean, T @ cov @ T.T

def warp_tracks(tracks, A):
    if np.allclose(A, np.eye(2, 3)):
        return
    for tr in tracks:
        tr.mean, tr.covariance = warp_state(tr.mean, tr.covariance, A)


# ---------------- ablation and reference estimators (added for the revision) ----------------
def fit_translation(pts):
    """Translation-only ablation: median displacement of the tracked corners, same gates as fit()
    (>= MIN_POINTS tracked, inlier ratio >= MIN_INLIER_RATIO within RANSAC_PX of the median)."""
    ident = np.eye(2, 3)
    if pts is None or len(pts[0]) < MIN_POINTS:
        return ident, 0 if pts is None else len(pts[0]), 0.0, True
    a, b = pts
    d = b - a
    med = np.median(d, axis=0)
    ratio = float((np.linalg.norm(d - med, axis=1) <= RANSAC_PX * SCALE).mean())
    if ratio < MIN_INLIER_RATIO:
        return ident, len(a), ratio, True
    A = ident.copy(); A[:, 2] = med / SCALE
    return A, len(a), ratio, False

ECC_SCALE, ECC_ITERS, ECC_EPS = 0.1, 100, 1e-5   # StrongSORT's ECC settings (Du et al., 2023)

def estimate_ecc(prev_img, img):
    """Reference estimator: Euclidean ECC alignment (Evangelidis and Psarakis, 2008) on grayscale
    frames downscaled to ECC_SCALE, as used by StrongSORT. No person mask. Returns (A, iters, cc, fallback)."""
    g0 = cv2.resize(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), None, fx=ECC_SCALE, fy=ECC_SCALE,
                    interpolation=cv2.INTER_AREA).astype(np.float32)
    g1 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=ECC_SCALE, fy=ECC_SCALE,
                    interpolation=cv2.INTER_AREA).astype(np.float32)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(g0, g1, W, cv2.MOTION_EUCLIDEAN,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS), None, 1)
    except cv2.error:
        return np.eye(2, 3), 0, 0.0, True
    A = W.astype(np.float64).copy(); A[:, 2] /= ECC_SCALE
    return A, ECC_ITERS, float(cc), False

if __name__ == "__main__":
    # 1) state warp: pure translation moves only the centre
    m = np.array([100., 50., .5, 80., 1., 2., 0., .1]); P = np.eye(8)
    A = np.array([[1., 0., 7.], [0., 1., -3.]])
    m2, P2 = warp_state(m, P, A)
    assert np.allclose(m2, [107, 47, .5, 80, 1, 2, 0, .1]) and np.allclose(P2, P)
    # 2) scale 2 doubles centre, height and their velocities, keeps aspect
    m3, _ = warp_state(m, P, np.array([[2., 0., 0.], [0., 2., 0.]]))
    assert np.allclose(m3, [200, 100, .5, 160, 2, 4, 0, .2])
    # 3) estimator recovers a known shift on a textured image, people masked or not
    import skimage.data
    img = cv2.cvtColor(skimage.data.coffee(), cv2.COLOR_RGB2BGR)
    big = cv2.resize(img, (1200, 800))
    f0, f1 = big[100:700, 100:1100], big[100 + 6:700 + 6, 100 - 10:1100 - 10]  # camera moves -> content +10, -6
    A, n, r, fb = estimate(to_gray(f0), to_gray(f1), [[400, 200, 100, 200]])
    assert not fb and abs(A[0, 2] - 10) < 1.0 and abs(A[1, 2] + 6) < 1.0, (A, n, r)
    # 4) oracle matches the same convention
    offs = np.array([[100, 100], [90, 106]])
    assert np.allclose(oracle(offs, 2)[:, 2], [10, -6])
    # 5) a blank frame forces the fallback
    blank = np.zeros((600, 1000, 3), np.uint8)
    assert estimate(to_gray(blank), to_gray(blank), [])[3]
    # 6) translation-only and ECC recover the same known shift
    At, _, _, fbt = fit_translation(track_points(to_gray(f0), to_gray(f1), [[400, 200, 100, 200]]))
    assert not fbt and abs(At[0, 2] - 10) < 1.0 and abs(At[1, 2] + 6) < 1.0, At
    Ae, _, cc, fbe = estimate_ecc(f0, f1)
    assert not fbe and abs(Ae[0, 2] - 10) < 1.5 and abs(Ae[1, 2] + 6) < 1.5, (Ae, cc)
    print("cmc self-check passed", A.round(3).tolist(), n, round(r, 3))
