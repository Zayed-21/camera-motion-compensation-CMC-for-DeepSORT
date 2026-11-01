"""Shared pieces: frame sources (real or synthetic crop), YOLOv8 person detector,
DeepSORT appearance encoder. Everything runs on OpenCV DNN + numpy (CPU)."""
import configparser, os
import cv2
import numpy as np
import reid_ref

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")          # data/MOT17/train/<seq>/{img1,gt,seqinfo.ini}

# ---------------- frames ----------------
def seqinfo(seq_dir):
    c = configparser.ConfigParser(); c.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = c["Sequence"]
    return dict(name=s["name"], fps=float(s["frameRate"]), n=int(s["seqLength"]),
                w=int(s["imWidth"]), h=int(s["imHeight"]), ext=s.get("imExt", ".jpg"))

class Frames:
    """Frame i (1-based) of a real sequence, or of a synthetic moving crop of it.
    offsets: None, or int array (n, 2) of crop top-left (x, y); crop size (cw, ch)."""
    def __init__(self, seq_dir, offsets=None, crop=None, step=1):
        """step > 1: keep every step-th frame (frames 1, 1+step, ...), renumbered 1..n."""
        self.dir, self.info = seq_dir, seqinfo(seq_dir)
        self.offsets, self.crop, self.step = offsets, crop, step
        self.n = -(-self.info["n"] // step)

    def __call__(self, i):
        i = (i - 1) * self.step + 1
        img = cv2.imread(os.path.join(self.dir, "img1", f"{i:06d}{self.info['ext']}"))
        if self.offsets is None:
            return img
        x, y = self.offsets[i - 1]; cw, ch = self.crop
        return img[y:y + ch, x:x + cw]

# ---------------- detector ----------------
class PersonDetector:
    """YOLOv8n (COCO) exported to ONNX; letterbox to 640 as Ultralytics does
    (pad value 114, RGB, /255). Keeps class 0 (person)."""
    def __init__(self, conf=0.25, iou=0.8, size=640):  # capstone inference settings (thesis p. 49)
        self.net = cv2.dnn.readNet(os.path.join(MODELS, "yolov8n.onnx"))
        self.conf, self.iou, self.size = conf, iou, size

    def __call__(self, img):
        h, w = img.shape[:2]; s = self.size
        r = min(s / h, s / w); nh, nw = round(h * r), round(w * r)
        top, left = (s - nh) // 2, (s - nw) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (s, s), swapRB=True))
        out = self.net.forward()[0].T                      # (8400, 84): cx, cy, w, h, 80 scores
        score = out[:, 4]
        keep = score >= self.conf
        out, score = out[keep], score[keep]
        if not len(out):
            return np.zeros((0, 5), np.float32)
        xywh = np.c_[out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]]
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        b = xywh[idx]
        b[:, 0] = (b[:, 0] - left) / r; b[:, 1] = (b[:, 1] - top) / r; b[:, 2:] /= r
        # clip to image (Ultralytics clips boxes too)
        x1 = np.clip(b[:, 0], 0, w); y1 = np.clip(b[:, 1], 0, h)
        x2 = np.clip(b[:, 0] + b[:, 2], 0, w); y2 = np.clip(b[:, 1] + b[:, 3], 0, h)
        return np.c_[x1, y1, x2 - x1, y2 - y1, score[idx]].astype(np.float32)

# ---------------- appearance encoder ----------------
def extract_image_patch(image, bbox, patch_shape=(128, 64)):
    """Copied from deep_sort/tools/generate_detections.py (Wojke et al., GPL-3.0)."""
    bbox = np.array(bbox, dtype=np.float64)
    target_aspect = float(patch_shape[1]) / patch_shape[0]
    new_width = target_aspect * bbox[3]
    bbox[0] -= (new_width - bbox[2]) / 2
    bbox[2] = new_width
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    return cv2.resize(image[sy:ey, sx:ex], tuple(patch_shape[::-1]))

class Encoder:
    """DeepSORT's released mars-small128 network. OpenCV runs it up to fc1/Elu (verified
    against a numpy interpreter of the full graph, max abs diff 4.8e-7); the final
    'ball' batch-norm + L2 normalisation is evaluated by the same numpy interpreter."""
    def __init__(self):
        self.net = cv2.dnn.readNetFromTensorflow(os.path.join(MODELS, "mars-small128-cv-fc1.pb"))
        self.full = os.path.join(MODELS, "mars-small128-cv.pb")
        self.tail = reid_ref.Tail(self.full, "fc1/Elu")

    def __call__(self, img, boxes):
        if not len(boxes):
            return np.zeros((0, 128), np.float32)
        patches = []
        for b in boxes:
            p = extract_image_patch(img, b)
            if p is None:  # same fallback as deep_sort: random patch
                p = np.random.default_rng(0).uniform(0, 255, (128, 64, 3)).astype(np.uint8)
            patches.append(p)
        self.net.setInput(cv2.dnn.blobFromImages(patches, 1.0, (64, 128), swapRB=True))
        return self.tail(self.net.forward()).astype(np.float32)
