"""
possession.py — Possession Estimation Engine for FastAPI & Streamlit
Handles: Spectator filtering, Referee classification, Goalkeeper protection,
         IoU tracking, EMA probability smoothing, annotated video output.
"""

import os
import sys
import glob
import random
from collections import Counter

import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
PERSON_CLASS_ID = 0
BALL_CLASS_ID   = 32
DETECT_CLASSES  = [0, 32]

TEAM_LABELS = {0: "Team A", 1: "Team B", 2: "Referee", -1: "Contested"}
TEAM_COLORS_BGR = {
    0: (80,  80, 255),    # Team A — Red
    1: (255, 120,  80),   # Team B — Blue
    2: (0,  225, 255),    # Referee — Gold/Yellow
    -1: (200, 200, 200),  # Contested
}

_yolo_model = None

# ─────────────────────────────────────────
# YOLO Model Loader (lazy, singleton)
# ─────────────────────────────────────────
def get_yolo_model(weights_name: str = "yolo11m.pt") -> YOLO:
    """
    Load YOLO model. Ultralytics auto-resolves the weights path:
    - Checks YOLO_CONFIG_DIR (set to /app/.yolo in Dockerfile)
    - Falls back to ~/.config/Ultralytics/
    - Downloads automatically if not found (dev/local only)
    On Railway the weights are baked into the image at build time.
    """
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    # Try nano as a lighter fallback if medium isn't cached
    for w in [weights_name, "yolo11n.pt", "yolov8m.pt", "yolov8n.pt"]:
        try:
            _yolo_model = YOLO(w)
            print(f"[Possession] Loaded YOLO weights: {w}")
            break
        except Exception:
            continue

    if _yolo_model is None:
        raise RuntimeError("Could not load any YOLO weights. Check the Dockerfile RUN step.")

    return _yolo_model


# ─────────────────────────────────────────
# Pitch Boundary Detection
# ─────────────────────────────────────────
def get_pitch_hull(frame):
    """Return convex hull of pitch grass region + binary green mask."""
    if frame is None or frame.size == 0:
        return None, None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([85, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    return hull, mask


def is_on_pitch(frame, box, pitch_hull=None, green_mask=None) -> bool:
    """Returns True when a detected person's feet land on the pitch (not in stands)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    foot = (float((x1 + x2) / 2.0), float(y2))

    if pitch_hull is not None and cv2.pointPolygonTest(pitch_hull, foot, True) < -15:
        return False

    if green_mask is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([30, 25, 25]), np.array([85, 255, 255]))

    box_h = max(1, y2 - y1)
    fy1 = max(0, int(y2 - 0.25 * box_h))
    fy2 = min(frame.shape[0], int(y2 + 0.05 * box_h))
    fx1, fx2 = max(0, x1), min(frame.shape[1], x2)
    roi = green_mask[fy1:fy2, fx1:fx2]
    if roi.size > 0 and np.mean(roi > 0) < 0.10:
        return False
    return True


# ─────────────────────────────────────────
# Dual Torso + Shorts HSV Feature Extractor
# ─────────────────────────────────────────
def _crop(frame, box, y_min_frac, y_max_frac):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    sy1 = max(0, y1 + int(h * y_min_frac))
    sy2 = min(frame.shape[0], max(sy1 + 1, y1 + int(h * y_max_frac)))
    sx1 = max(0, x1 + int(w * 0.12))
    sx2 = min(frame.shape[1], max(sx1 + 1, x2 - int(w * 0.12)))
    return frame[sy1:sy2, sx1:sx2]


def _hsv_hist(crop, bins_h=12, bins_s=6):
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h >= 35) & (h <= 85)
    mask = (s > 25) & (v > 25) & (v < 250) & (~grass)
    if mask.sum() < 10:
        mask = (s > 15) & (v > 15) & (v < 250)
    if mask.sum() < 5:
        mask = np.ones_like(mask, dtype=bool)
    h_h, _ = np.histogram(h[mask], bins=bins_h, range=(0, 180), density=True)
    s_h, _ = np.histogram(s[mask], bins=bins_s, range=(0, 256), density=True)
    return np.concatenate([h_h, s_h]).astype(np.float32)


def jersey_color_hist(frame, box):
    """Dual HSV histogram: Torso (shirt) + Shorts."""
    t_hist = _hsv_hist(_crop(frame, box, 0.15, 0.48))
    s_hist = _hsv_hist(_crop(frame, box, 0.45, 0.75))
    if t_hist is None and s_hist is None:
        return None
    if t_hist is None:
        return np.concatenate([s_hist, s_hist])
    if s_hist is None:
        return np.concatenate([t_hist, t_hist])
    return np.concatenate([t_hist, s_hist]).astype(np.float32)


# ─────────────────────────────────────────
# IoU Tracker
# ─────────────────────────────────────────
def _iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


class _Track:
    def __init__(self, tid, box, probs):
        self.tid = tid
        self.box = box
        self.probs = np.array(probs, dtype=float)
        self.missed = 0

    def update(self, box, probs, alpha=0.70):
        self.box = box
        self.probs = alpha * self.probs + (1 - alpha) * np.array(probs, dtype=float)
        self.missed = 0


def _foot_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, float(y2)


# ─────────────────────────────────────────
# Team Classifier Builder
# ─────────────────────────────────────────
def build_team_classifier(detections: list, match_name: str = "Match"):
    """
    Fits a 3-class MLP classifier (Team A / Team B / Referee).
    Goalkeepers near goal areas are protected and assigned to their team.
    Returns classifier dict or None if too few samples.
    """
    if len(detections) < 10:
        return None

    feats, meta = [], []
    for d in detections:
        f = jersey_color_hist(d["frame_img"], d["box"])
        if f is not None:
            feats.append(f)
            meta.append(d)
    if len(feats) < 15:
        return None

    F = np.stack(feats)

    # Find 2 dominant team centroids
    dists = euclidean_distances(F)
    density = (dists < np.median(dists)).sum(axis=1)
    a1 = int(np.argmax(density))
    a2 = int(np.argmax(dists[a1] * (density > np.percentile(density, 30))))
    c1, c2 = F[a1].copy(), F[a2].copy()

    for _ in range(3):
        d1 = np.linalg.norm(F - c1, axis=1)
        d2 = np.linalg.norm(F - c2, axis=1)
        if (d1 <= d2).sum(): c1 = F[d1 <= d2].mean(0)
        if (d2 < d1).sum():  c2 = F[d2 < d1].mean(0)

    d1 = np.linalg.norm(F - c1, axis=1)
    d2 = np.linalg.norm(F - c2, axis=1)
    min_d    = np.minimum(d1, d2)
    rel_diff = np.abs(d1 - d2) / (min_d + 1e-5)

    pseudo = np.where(d1 <= d2, 0, 1)
    outlier_mask = (min_d > np.percentile(min_d, 78)) | \
                   ((rel_diff < 0.08) & (min_d > np.median(min_d)))

    half = F.shape[1] // 2
    c1s, c2s = c1[half:], c2[half:]

    for idx in np.where(outlier_mask)[0]:
        box   = meta[idx]["box"]
        img   = meta[idx].get("frame_img")
        img_w = img.shape[1] if img is not None else 1280
        xc    = (box[0] + box[2]) / 2.0
        # Goalkeeper zone: within 18% of either side
        if xc < 0.18 * img_w or xc > 0.82 * img_w:
            sf = F[idx, half:]
            pseudo[idx] = 0 if np.linalg.norm(sf - c1s) <= np.linalg.norm(sf - c2s) else 1
        else:
            pseudo[idx] = 2   # Referee

    strat = pseudo if len(np.unique(pseudo)) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        F, pseudo, test_size=0.25, random_state=42, stratify=strat)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation="relu", solver="adam",
            max_iter=5000, early_stopping=True, n_iter_no_change=60,
            random_state=42,
        )),
    ])
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    return {
        "match":       match_name,
        "classifier":  clf,
        "val_acc":     float(acc),
        "n_team_a":    int((pseudo == 0).sum()),
        "n_team_b":    int((pseudo == 1).sum()),
        "n_referee":   int((pseudo == 2).sum()),
    }


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────
def process_video_possession(
    video_path:       str,
    output_video_path: str,
    max_frames:       int   = 300,
    conf_thresh:      float = 0.20,
    ball_conf_thresh: float = 0.10,
    poss_radius_px:   float = 100.0,
    smoothing_window: int   = 5,
    yolo_weights:     str   = "yolo11m.pt",
    det_imgsz:        int   = 960,
) -> dict:
    """
    Full possession pipeline:
    Phase 1 → Sample frames to fit team classifier
    Phase 2 → Frame-by-frame inference + IoU tracking + annotated video output

    Returns a dict with possession stats, per-frame records, and output video path.
    """
    model = get_yolo_model(yolo_weights)
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w  = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    # ── Phase 1: Sample frames for classifier ──
    step        = max(1, int(fps * 2.0))
    sample_idxs = list(range(0, min(total, max_frames), step))
    sample_dets = []

    for si in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
        ret, frame = cap.read()
        if not ret:
            continue
        hull, gmask = get_pitch_hull(frame)
        res = model.predict(frame, conf=conf_thresh, classes=DETECT_CLASSES,
                            imgsz=det_imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        for box, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                     res.boxes.cls.cpu().numpy().astype(int),
                                     res.boxes.conf.cpu().numpy()):
            if cls_id == PERSON_CLASS_ID and conf >= conf_thresh:
                if is_on_pitch(frame, box, hull, gmask):
                    sample_dets.append({"frame": si, "box": box.tolist(), "frame_img": frame})

    tc = build_team_classifier(sample_dets)
    clf     = tc["classifier"] if tc else None
    val_acc = tc["val_acc"]    if tc else 0.0

    def _get_probs(feat):
        if clf is None:
            return np.array([0.5, 0.5, 0.0])
        if hasattr(clf, "predict_proba"):
            raw = clf.predict_proba([feat])[0]
            classes = getattr(clf, "classes_", [0, 1, 2])
            p = np.zeros(3, dtype=float)
            for ci, cv_ in enumerate(classes):
                if cv_ in (0, 1, 2):
                    p[cv_] = raw[ci]
            return p
        pred = clf.predict([feat])[0]
        p = np.zeros(3)
        p[pred] = 1.0
        return p

    # ── Phase 2: Inference ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx       = 0
    past_poss       = []
    recent_poss     = []
    per_frame       = []
    active_tracks   = []
    next_tid        = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        hull, gmask = get_pitch_hull(frame)
        res = model.predict(frame, conf=min(conf_thresh, ball_conf_thresh),
                            classes=DETECT_CLASSES, imgsz=det_imgsz, verbose=False)[0]

        det_boxes, det_probs = [], []
        ball_pos, best_bconf = None, 0.0

        if res.boxes is not None and len(res.boxes):
            for box, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(),
                                         res.boxes.cls.cpu().numpy().astype(int),
                                         res.boxes.conf.cpu().numpy()):
                if cls_id == PERSON_CLASS_ID and conf >= conf_thresh:
                    if not is_on_pitch(frame, box, hull, gmask):
                        continue
                    feat = jersey_color_hist(frame, box)
                    if feat is None:
                        continue
                    det_boxes.append(box)
                    det_probs.append(_get_probs(feat))
                elif cls_id == BALL_CLASS_ID and conf >= ball_conf_thresh:
                    if conf > best_bconf:
                        best_bconf = conf
                        ball_pos = ((box[0]+box[2])/2, (box[1]+box[3])/2)

        # IoU matching
        updated, matched = [], set()
        for trk in active_tracks:
            bi, iou_best = -1, 0.0
            for di, db in enumerate(det_boxes):
                if di in matched:
                    continue
                v = _iou(trk.box, db)
                if v > iou_best:
                    iou_best, bi = v, di
            if iou_best > 0.3 and bi != -1:
                trk.update(det_boxes[bi], det_probs[bi])
                matched.add(bi)
                updated.append(trk)
            else:
                trk.missed += 1
                if trk.missed <= 10:
                    updated.append(trk)

        for di, (db, dp) in enumerate(zip(det_boxes, det_probs)):
            if di not in matched:
                updated.append(_Track(next_tid, db, dp))
                next_tid += 1

        active_tracks = updated
        p_boxes = [t.box  for t in active_tracks if t.missed == 0]
        p_teams = [int(np.argmax(t.probs)) for t in active_tracks if t.missed == 0]

        # Possession
        cur_poss, best_d = -1, float("inf")
        if ball_pos:
            for bx, tm in zip(p_boxes, p_teams):
                if tm == 2:
                    continue
                fx, fy = _foot_center(bx)
                d = ((ball_pos[0]-fx)**2 + (ball_pos[1]-fy)**2)**0.5
                if d < best_d:
                    best_d, cur_poss = d, tm
            if best_d > poss_radius_px:
                cur_poss = -1

        recent_poss.append(cur_poss)
        if len(recent_poss) > smoothing_window:
            recent_poss.pop(0)
        valid = [p for p in recent_poss if p in (0, 1)]
        smooth = Counter(valid).most_common(1)[0][0] if valid else cur_poss
        if smooth in (0, 1):
            past_poss.append(smooth)

        claimed  = [p for p in past_poss if p in (0, 1)]
        n_cl     = len(claimed)
        pct_a = round(100.0 * claimed.count(0) / n_cl, 1) if n_cl else 50.0
        pct_b = round(100.0 * claimed.count(1) / n_cl, 1) if n_cl else 50.0

        per_frame.append({
            "frame":          frame_idx,
            "time_sec":       round(frame_idx / fps, 2),
            "possession":     smooth,
            "possession_label": TEAM_LABELS.get(smooth, "Contested"),
            "team_a_pct":     pct_a,
            "team_b_pct":     pct_b,
        })

        # Draw annotations
        for bx, tm in zip(p_boxes, p_teams):
            x1, y1, x2, y2 = [int(v) for v in bx]
            col = TEAM_COLORS_BGR.get(tm, (200, 200, 200))
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
            cv2.putText(frame, TEAM_LABELS.get(tm,"?"), (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(frame, (cx,cy), 12, (0,255,0), 3)
            cv2.putText(frame, "Ball", (cx+14,cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # HUD
        cv2.rectangle(frame, (0,0), (W,65), (20,20,20), -1)
        bm  = 20
        bw  = max(10, W - 2*bm)
        baw = int(bw * pct_a / 100.0)
        cv2.rectangle(frame, (bm,42), (bm+baw,54),    (80,80,255), -1)
        cv2.rectangle(frame, (bm+baw,42),(bm+bw,54),  (255,120,80),-1)
        cv2.putText(frame,
            f"Possession:  Team A {pct_a:.1f}%   |   Team B {pct_b:.1f}%",
            (bm,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        out_w.write(frame)
        frame_idx += 1

    cap.release()
    out_w.release()

    ca = sum(1 for r in per_frame if r["possession"] == 0)
    cb = sum(1 for r in per_frame if r["possession"] == 1)
    nc = ca + cb

    return {
        "status":                   "success",
        "total_frames_processed":   frame_idx,
        "fps":                      float(fps),
        "team_a_pct":               round(100.0*ca/nc, 1) if nc else 50.0,
        "team_b_pct":               round(100.0*cb/nc, 1) if nc else 50.0,
        "team_a_frames":            ca,
        "team_b_frames":            cb,
        "n_team_a":                 tc["n_team_a"]   if tc else 0,
        "n_team_b":                 tc["n_team_b"]   if tc else 0,
        "n_referee":                tc["n_referee"]  if tc else 0,
        "classifier_val_acc":       val_acc,
        "per_frame_records":        per_frame,
        "output_video_filename":    os.path.basename(output_video_path),
    }
