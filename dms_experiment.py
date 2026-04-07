#!/usr/bin/env python3
"""
DMS-Raptor: Dynamic Model Switching for Real-Time UAV Inspection
================================================================
Complete Experiment Pipeline -- Single File

Author:  Ganapathi Teja Chirravuri
University: Technische Universitat Chemnitz, Faculty of Computer Science

Usage:
    python dms_experiment.py --help
    python dms_experiment.py run-all    --device cpu --max-frames 500
    python dms_experiment.py run-all    --device cuda
    python dms_experiment.py detection  --device cpu
    python dms_experiment.py timing     --device cpu --trials 3
    python dms_experiment.py validation --device cuda --samples 30
    python dms_experiment.py plots      --results-dir results
    python dms_experiment.py report     --results-dir results

Sections:
    1. Configuration & Data Classes
    2. NR-IQA Score (BRISQUE-like)
    3. Image Proxies (Laplacian, Entropy, Tenengrad, Color Entropy)
    4. Streaming Engine with all 8 Policies
    5. Detection Quality Analysis (dual-model, s_only as oracle)
    6. Repeated Trials (timing statistics)
    7. Frame-Level Validation (30 samples/policy, side-by-side images)
    8. Plot Generation (50+ charts, bars, comparisons)
    9. Thesis Report Generator (complete analysis)
   10. CLI Entry Point
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

# Optional imports (graceful fallback)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.ndimage import gaussian_filter
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


# =====================================================================
# 1. CONFIGURATION & DATA CLASSES
# =====================================================================

POLICIES = [
    "n_only", "s_only", "entropy_only", "combined", "combined_hyst",
    "conf_ema", "niqe_switch", "multi_proxy",
]

SWITCHING_POLICIES = [
    "entropy_only", "combined", "combined_hyst",
    "conf_ema", "niqe_switch", "multi_proxy",
]

POLICY_COLORS = {
    "n_only":        "#2196F3",  # blue
    "s_only":        "#F44336",  # red
    "entropy_only":  "#4CAF50",  # green
    "combined":      "#FF9800",  # orange
    "combined_hyst": "#9C27B0",  # purple
    "conf_ema":      "#00BCD4",  # cyan
    "niqe_switch":   "#795548",  # brown
    "multi_proxy":   "#607D8B",  # blue-grey
}

POLICY_LABELS = {
    "n_only":        "n_only (YOLOv8n)",
    "s_only":        "s_only (YOLOv8s)",
    "entropy_only":  "entropy_only",
    "combined":      "combined",
    "combined_hyst": "combined_hyst",
    "conf_ema":      "conf_ema",
    "niqe_switch":   "niqe_switch",
    "multi_proxy":   "multi_proxy",
}


@dataclass
class InferenceParams:
    """Hardware and inference settings."""
    imgsz: int = 640
    device: str = "cpu"
    conf_min: float = 0.001
    iou_nms: float = 0.45
    conf_show: float = 0.25
    max_frames: int = 0   # 0 = all frames
    stride: int = 1


@dataclass
class RunConfig:
    """Controller configuration for a single policy run."""
    name: str = ""
    policy: str = "combined_hyst"
    mode: str = "fixed"

    # Scene proxy weighting
    alpha: float = 0.6

    # Fixed thresholds
    c_low: float = 0.45
    c_high: float = 0.55
    combined_mid: float = 0.50

    # Rolling window
    history_window_size: int = 200
    probe_every_k: int = 1
    latency_smooth_window: int = 15

    # Adaptive percentile bounds
    norm_lo: float = 10.0
    norm_hi: float = 90.0
    thr_lo: float = 35.0
    thr_hi: float = 65.0
    thr_mid: float = 50.0

    # Latency budget (adaptive mode)
    latency_budget_ms: float = 150.0
    latency_penalty_factor: float = 0.01
    budget_guard_margin: float = 0.10
    budget_guard_frames: int = 8

    # Actuator stabilizers
    c_ema_beta: float = 0.25
    min_dwell_frames: int = 10
    max_switches_per_100: int = 12

    # Proxy computation
    proxy_size: int = 160
    hist_bins: int = 64
    downsample: int = 8

    # conf_ema policy
    conf_ema_fast_beta: float = 0.30
    conf_ema_slow_beta: float = 0.02
    conf_ema_c_high: float = 0.12
    conf_ema_c_low: float = 0.04

    # niqe_switch policy
    niqe_fast_beta: float = 0.30
    niqe_slow_beta: float = 0.02
    niqe_c_high: float = 0.003
    niqe_c_low: float = 0.001

    # multi_proxy policy (cross-validated weights)
    mp_w_laplacian: float = 0.37
    mp_w_entropy: float = 0.29
    mp_w_tenengrad: float = 0.38
    mp_w_edge_density: float = 0.0
    mp_w_local_contrast: float = 0.0
    mp_w_brenner: float = 0.0
    mp_w_color_entropy: float = 0.28
    mp_fast_beta: float = 0.30
    mp_slow_beta: float = 0.02
    mp_c_high: float = 0.10
    mp_c_low: float = 0.03

    # Zero-detection gate
    zero_det_gate: bool = False
    zero_det_lookback: int = 3
    zero_det_conf_thresh: float = 0.25

    # Output
    save_annotated_video: bool = False
    output_video_path: str = ""


@dataclass
class FrameResult:
    """Per-frame result from the streaming engine."""
    frame_idx: int = 0
    annotated_frame: Optional[np.ndarray] = None
    choice: str = "n"
    num_detections: int = 0
    L: float = 0.0
    H: float = 0.0
    C: float = 0.0
    c_low: float = 0.0
    c_high: float = 0.0
    penalty: float = 0.0
    avg_T_total: float = 0.0
    T_scene_ms: float = 0.0
    T_ctrl_ms: float = 0.0
    T_infer_n_ms: float = 0.0
    T_infer_s_ms: float = 0.0
    T_total_ms: float = 0.0
    dwell: int = 0
    budget_guard_left: int = 0
    mean_conf: float = 0.0
    conf_drop: float = 0.0
    niqe_score: float = 0.0
    zero_det_gated: bool = False


@dataclass
class RunSummary:
    """Aggregated summary after a full run."""
    video: str = ""
    run_name: str = ""
    policy: str = ""
    mode: str = ""
    total_frames: int = 0

    T_scene_ms_mean: float = 0.0
    T_ctrl_ms_mean: float = 0.0
    T_infer_n_ms_mean: float = 0.0
    T_infer_s_ms_mean: float = 0.0
    T_infer_n_ms_cond_mean: float = 0.0
    T_infer_s_ms_cond_mean: float = 0.0
    T_total_ms_mean: float = 0.0
    T_total_ms_p95: float = 0.0
    T_total_ms_p99: float = 0.0
    slow_pct: float = 0.0
    switches: int = 0
    sw_per_100: float = 0.0
    zero_det_gate_activations: int = 0

    run_timestamp: str = ""
    output_video_path: str = ""
    output_video_size_mb: float = 0.0

    # Full per-frame traces
    frame_indices: List[int] = field(default_factory=list)
    T_total_trace: List[float] = field(default_factory=list)
    T_scene_trace: List[float] = field(default_factory=list)
    T_ctrl_trace: List[float] = field(default_factory=list)
    T_infer_n_trace: List[float] = field(default_factory=list)
    T_infer_s_trace: List[float] = field(default_factory=list)
    C_trace: List[float] = field(default_factory=list)
    L_trace: List[float] = field(default_factory=list)
    H_trace: List[float] = field(default_factory=list)
    choice_trace: List[str] = field(default_factory=list)
    c_low_trace: List[float] = field(default_factory=list)
    c_high_trace: List[float] = field(default_factory=list)
    dwell_trace: List[int] = field(default_factory=list)
    num_detections_trace: List[int] = field(default_factory=list)
    penalty_trace: List[float] = field(default_factory=list)
    avg_T_total_trace: List[float] = field(default_factory=list)
    mean_conf_trace: List[float] = field(default_factory=list)
    conf_drop_trace: List[float] = field(default_factory=list)
    niqe_trace: List[float] = field(default_factory=list)
    zero_det_gated_trace: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                d[f.name] = val.tolist()
            elif isinstance(val, (np.floating, np.integer)):
                d[f.name] = float(val)
            else:
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSummary":
        valid = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# =====================================================================
# 2. NR-IQA SCORE (BRISQUE-like)
# =====================================================================

def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if HAS_SCIPY:
        return gaussian_filter(img, sigma=sigma)
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _estimate_ggd_shape(x: np.ndarray) -> float:
    x = x.ravel().astype(np.float64)
    if len(x) < 10:
        return 2.0
    sigma = float(np.std(x))
    if sigma < 1e-7:
        return 10.0
    mean_abs = float(np.mean(np.abs(x)))
    if mean_abs < 1e-7:
        return 10.0
    r = (sigma * sigma) / (mean_abs * mean_abs)
    beta = max(0.2, min(10.0, 1.0 / (r - 0.3189 + 1e-6) * 0.7894))
    return float(np.clip(beta, 0.2, 10.0))


def _compute_mscn(gray: np.ndarray, sigma: float = 7.0 / 6.0) -> np.ndarray:
    mu = _gaussian_blur(gray, sigma)
    mu_sq = _gaussian_blur(gray * gray, sigma)
    sigma_map = np.sqrt(np.maximum(mu_sq - mu * mu, 0.0)) + 1.0 / 255.0
    return (gray - mu) / sigma_map


def compute_nriqa_score(img_bgr: np.ndarray, proxy_size: int = 160) -> float:
    """Lightweight NR-IQA quality score. Higher = worse quality."""
    if proxy_size > 0:
        img_small = cv2.resize(img_bgr, (proxy_size, proxy_size),
                               interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY).astype(np.float64)
    scores = []
    for scale in (1, 2):
        if scale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (max(16, w // 2), max(16, h // 2)),
                              interpolation=cv2.INTER_AREA)
        mscn = _compute_mscn(gray)
        beta = _estimate_ggd_shape(mscn)
        mscn_var = float(np.var(mscn))
        beta_h = _estimate_ggd_shape(mscn[:, :-1] * mscn[:, 1:])
        beta_v = _estimate_ggd_shape(mscn[:-1, :] * mscn[1:, :])
        scale_score = (
            (10.0 - beta) * 3.0 +
            (10.0 - beta_h) * 2.0 +
            (10.0 - beta_v) * 2.0 +
            max(0.0, 1.0 - mscn_var) * 5
        )
        scores.append(max(0.0, scale_score))
    return float(np.mean(scores))


# =====================================================================
# 3. IMAGE PROXIES
# =====================================================================

def complexity_proxies_fast(img_bgr: np.ndarray, proxy_size: int = 160,
                            hist_bins: int = 64) -> Tuple[float, float]:
    """Compute Laplacian variance (L) and Shannon entropy (H)."""
    img_small = cv2.resize(img_bgr, (proxy_size, proxy_size),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    L = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    hist = cv2.calcHist([gray], [0], None, [hist_bins], [0, 256]).astype(np.float32).ravel()
    s = float(hist.sum())
    if s <= 0:
        return L, 0.0
    p = hist[hist > 1e-12] / s
    H = float(-(p * np.log2(p)).sum())
    return L, H


def compute_extended_proxies(gray: np.ndarray) -> dict:
    """Tenengrad, EdgeDensity, LocalContrast, Brenner."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float((gx ** 2 + gy ** 2).mean())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.astype(np.float32).mean() / 255.0)
    local_contrast = float(gray.astype(np.float32).std())
    shifted = np.zeros_like(gray, dtype=np.float32)
    shifted[:, :-2] = gray[:, 2:].astype(np.float32)
    brenner = float(((shifted - gray.astype(np.float32)) ** 2).mean())
    return {"tenengrad": tenengrad, "edge_density": edge_density,
            "local_contrast": local_contrast, "brenner": brenner}


def compute_color_entropy(img_bgr_small: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy on joint H,S histogram (HSV)."""
    hsv = cv2.cvtColor(img_bgr_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins],
                        [0, 180, 0, 256]).astype(np.float32).ravel()
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist[hist > 1e-12] / s
    return float(-(p * np.log2(p)).sum())


# =====================================================================
# 4. STREAMING ENGINE WITH ALL 8 POLICIES
# =====================================================================

class FastRollingPercentile:
    """O(N) insert, O(1) query rolling percentile."""
    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self.history: deque = deque(maxlen=self.maxlen)
        self.sorted_list: List[float] = []

    def add(self, val: float):
        val = float(val)
        if len(self.history) == self.maxlen:
            oldest = self.history.popleft()
            idx = bisect.bisect_left(self.sorted_list, oldest)
            if 0 <= idx < len(self.sorted_list) and self.sorted_list[idx] == oldest:
                self.sorted_list.pop(idx)
            else:
                try:
                    self.sorted_list.remove(oldest)
                except ValueError:
                    pass
        self.history.append(val)
        bisect.insort(self.sorted_list, val)

    def percentile(self, p: float) -> float:
        if not self.sorted_list:
            return 0.0
        p = float(np.clip(p, 0.0, 100.0))
        idx = int((p / 100.0) * (len(self.sorted_list) - 1))
        return float(self.sorted_list[idx])

    def __len__(self):
        return len(self.history)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def run_model_single(model, img_bgr: np.ndarray, imgsz: int,
                     device: str, conf: float, iou_nms: float) -> List[np.ndarray]:
    """Run YOLO inference -> list of [x1,y1,x2,y2,conf]. No secondary NMS."""
    res = model.predict(source=img_bgr, imgsz=imgsz, device=device,
                        conf=conf, iou=iou_nms, verbose=False)[0]
    dets: List[np.ndarray] = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        for b, c in zip(xyxy, cf):
            dets.append(np.array([b[0], b[1], b[2], b[3], c], dtype=np.float32))
    return dets


def draw_overlay(frame: np.ndarray, dets: List[np.ndarray], choice: str,
                 header_lines: List[str], conf_show: float = 0.25) -> np.ndarray:
    """Draw annotated overlay: colored border + boxes + header panel."""
    out = frame.copy()
    fh, fw = out.shape[:2]
    accent = (80, 175, 76) if choice == "n" else (96, 69, 233)
    model_label = "YOLOv8n" if choice == "n" else "YOLOv8s"
    panel_bg = (26, 14, 10)

    # Border
    border = max(8, min(fh, fw) // 50)
    overlay_border = out.copy()
    cv2.rectangle(overlay_border, (0, 0), (fw - 1, fh - 1), accent, border)
    cv2.addWeighted(overlay_border, 0.75, out, 0.25, 0, out)

    # Boxes
    for d in dets:
        x1, y1, x2, y2, c = map(float, d)
        if c < conf_show:
            continue
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), accent, 2, cv2.LINE_AA)
        label = f"{c:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.5, 1)
        lbl_y1 = max(0, int(y1) - th - baseline - 4)
        overlay_lbl = out.copy()
        cv2.rectangle(overlay_lbl, (int(x1), lbl_y1), (int(x1) + tw + 6, max(0, int(y1))), accent, -1)
        cv2.addWeighted(overlay_lbl, 0.85, out, 0.15, 0, out)
        cv2.putText(out, label, (int(x1) + 3, max(0, int(y1)) - baseline - 1),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Header panel
    x0, y0 = border + 6, border + 6
    line_h = 24
    font = cv2.FONT_HERSHEY_DUPLEX
    box_w = min(700, fw - 2 * x0)
    box_h = line_h * (len(header_lines) + 1) + 8
    overlay_hdr = out.copy()
    cv2.rectangle(overlay_hdr, (x0, y0), (x0 + box_w, y0 + box_h), panel_bg, -1)
    cv2.addWeighted(overlay_hdr, 0.75, out, 0.25, 0, out)
    for i, s in enumerate(header_lines):
        cv2.putText(out, s, (x0 + 6, y0 + 14 + (i + 1) * line_h),
                    font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Badge
    (btw, bth), _ = cv2.getTextSize(model_label, font, 0.7, 2)
    bx1 = fw - btw - 28 - border
    by1 = border + 8
    overlay_badge = out.copy()
    cv2.rectangle(overlay_badge, (bx1, by1), (fw - border - 8, by1 + bth + 16), accent, -1)
    cv2.addWeighted(overlay_badge, 0.85, out, 0.15, 0, out)
    cv2.putText(out, model_label, (bx1 + 10, by1 + bth + 8),
                font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class StreamingEngine:
    """Generator-based inference engine. Runs ONE model per frame."""

    def __init__(self, source_path: str, cfg: RunConfig,
                 model_n, model_s, inf: InferenceParams):
        self.source_path = source_path
        self.cfg = cfg
        self.model_n = model_n
        self.model_s = model_s
        self.inf = inf
        self._stop_flag = False

        cap = cv2.VideoCapture(source_path)
        self._n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        if inf.max_frames > 0 and self._n_total > 0:
            self._n_total = min(self._n_total, inf.max_frames * max(1, inf.stride))
        if inf.stride > 1 and self._n_total > 0:
            self._n_total = self._n_total // inf.stride

        # Accumulators
        self._totals: List[float] = []
        self._t_scene: List[float] = []
        self._t_ctrl: List[float] = []
        self._t_in_n: List[float] = []
        self._t_in_s: List[float] = []
        self._L_trace: List[float] = []
        self._H_trace: List[float] = []
        self._dwell_trace: List[int] = []
        self._det_trace: List[int] = []
        self._penalty_trace: List[float] = []
        self._avg_total_trace: List[float] = []
        self._slow_frames = 0
        self._switches = 0
        self._frame_indices: List[int] = []
        self._C_trace: List[float] = []
        self._choice_trace: List[str] = []
        self._c_low_trace: List[float] = []
        self._c_high_trace: List[float] = []
        self._mean_conf_trace: List[float] = []
        self._conf_drop_trace: List[float] = []
        self._niqe_trace: List[float] = []
        self._zero_det_gated_trace: List[bool] = []
        self._summary: Optional[RunSummary] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._output_video_path: str = ""

    @property
    def total_frames(self) -> int:
        return max(1, self._n_total) if self._n_total > 0 else 0

    def stop(self):
        self._stop_flag = True

    def _init_video_writer(self, frame_shape, fps):
        if not self.cfg.save_annotated_video or not self.cfg.output_video_path:
            return
        h, w = frame_shape[:2]
        out_path = self.cfg.output_video_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if self._video_writer.isOpened():
            self._output_video_path = out_path
        else:
            self._video_writer = None

    def run(self) -> Generator[FrameResult, None, None]:
        cfg = self.cfg
        inf = self.inf

        rolling_L = FastRollingPercentile(cfg.history_window_size)
        rolling_H = FastRollingPercentile(cfg.history_window_size)
        rolling_C = FastRollingPercentile(cfg.history_window_size)
        latency_hist: deque = deque(maxlen=cfg.latency_smooth_window)
        switch_events: deque = deque()

        last_LH: Optional[Tuple[float, float]] = None
        last_LH_i: int = -(10 ** 9)
        last_choice: Optional[str] = None
        dwell = 0
        budget_guard_left = 0
        c_ema_state: Optional[float] = None

        # conf_ema state
        last_mean_conf: float = 0.0
        conf_ema_fast: Optional[float] = None
        conf_ema_slow: Optional[float] = None

        # niqe_switch state
        niqe_ema_fast: Optional[float] = None
        niqe_ema_slow: Optional[float] = None

        # multi_proxy EMA state
        mp_ema: dict = {}

        # Zero-detection gate
        recent_det_counts: deque = deque(maxlen=max(1, cfg.zero_det_lookback))
        zero_det_gated = False

        writer_initialized = False
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.source_path}")

        kept = 0
        raw_idx = 0
        while True:
            if self._stop_flag:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if inf.stride > 1 and (raw_idx % inf.stride) != 0:
                raw_idx += 1
                continue
            raw_idx += 1

            if not writer_initialized:
                self._init_video_writer(frame.shape, self._fps)
                writer_initialized = True

            # -- 1) SCENE ANALYSIS --
            T_scene_ms = 0.0
            L: Optional[float] = None
            H: Optional[float] = None
            C: Optional[float] = None
            Hmed = 0.0

            needs_scene = cfg.policy in ("entropy_only", "combined", "combined_hyst")
            needs_conf_ema = cfg.policy == "conf_ema"
            needs_niqe = cfg.policy == "niqe_switch"
            needs_multi = cfg.policy == "multi_proxy"
            conf_drop_val = 0.0

            if needs_scene:
                t0 = time.perf_counter()
                if (kept - last_LH_i) >= cfg.probe_every_k or last_LH is None:
                    L, H = complexity_proxies_fast(frame, cfg.proxy_size, cfg.hist_bins)
                    last_LH = (L, H)
                    last_LH_i = kept
                else:
                    L, H = last_LH
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            niqe_val_raw = 0.0
            if needs_niqe:
                t0 = time.perf_counter()
                niqe_val_raw = compute_nriqa_score(frame, cfg.proxy_size)
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            mp_raw: Optional[dict] = None
            if needs_multi:
                t0 = time.perf_counter()
                mp_small = cv2.resize(frame, (cfg.proxy_size, cfg.proxy_size),
                                      interpolation=cv2.INTER_AREA)
                mp_gray = cv2.cvtColor(mp_small, cv2.COLOR_BGR2GRAY)
                L = float(cv2.Laplacian(mp_gray, cv2.CV_32F).var())
                bins_ = int(max(16, cfg.hist_bins))
                hist_ = cv2.calcHist([mp_gray], [0], None, [bins_], [0, 256]).astype(np.float32).ravel()
                s_ = float(hist_.sum())
                H = 0.0
                if s_ > 0:
                    p_ = hist_[hist_ > 1e-12] / s_
                    H = float(-(p_ * np.log2(p_)).sum())
                ext = compute_extended_proxies(mp_gray)
                col_ent = compute_color_entropy(mp_small)
                mp_raw = {"L": L, "H": H, "tenengrad": ext["tenengrad"],
                          "edge_density": ext["edge_density"],
                          "local_contrast": ext["local_contrast"],
                          "brenner": ext["brenner"], "color_entropy": col_ent}
                T_scene_ms = (time.perf_counter() - t0) * 1000.0

            # -- 2) CONTROLLER --
            t0_ctrl = time.perf_counter()

            if needs_conf_ema:
                bf_c = cfg.conf_ema_fast_beta
                bs_c = cfg.conf_ema_slow_beta
                if conf_ema_fast is None:
                    conf_ema_fast = last_mean_conf
                    conf_ema_slow = last_mean_conf
                else:
                    conf_ema_fast = bf_c * last_mean_conf + (1.0 - bf_c) * conf_ema_fast
                    conf_ema_slow = bs_c * last_mean_conf + (1.0 - bs_c) * conf_ema_slow
                conf_drop_val = max(0.0, (conf_ema_slow - conf_ema_fast) / (conf_ema_slow + 1e-9))
                C = float(conf_drop_val)

            if needs_scene:
                rolling_L.add(L)
                rolling_H.add(H)
                if cfg.mode == "fixed":
                    L_lo, L_hi = rolling_L.percentile(0), rolling_L.percentile(100)
                    H_lo, H_hi = rolling_H.percentile(0), rolling_H.percentile(100)
                else:
                    L_lo, L_hi = rolling_L.percentile(cfg.norm_lo), rolling_L.percentile(cfg.norm_hi)
                    H_lo, H_hi = rolling_H.percentile(cfg.norm_lo), rolling_H.percentile(cfg.norm_hi)
                Ln = float(np.clip((L - L_lo) / (L_hi - L_lo + 1e-9), 0.0, 1.0))
                Hn = float(np.clip((H - H_lo) / (H_hi - H_lo + 1e-9), 0.0, 1.0))
                C_raw = float(cfg.alpha * Ln + (1.0 - cfg.alpha) * Hn)
                if cfg.c_ema_beta <= 0.0:
                    C = C_raw
                else:
                    if c_ema_state is None:
                        c_ema_state = C_raw
                    else:
                        c_ema_state = (1.0 - cfg.c_ema_beta) * c_ema_state + cfg.c_ema_beta * C_raw
                    C = float(c_ema_state)
                rolling_C.add(C)
                Hmed = rolling_H.percentile(50)

            if needs_niqe:
                bf_n, bs_n = cfg.niqe_fast_beta, cfg.niqe_slow_beta
                if niqe_ema_fast is None:
                    niqe_ema_fast = niqe_val_raw
                    niqe_ema_slow = niqe_val_raw
                else:
                    niqe_ema_fast = bf_n * niqe_val_raw + (1.0 - bf_n) * niqe_ema_fast
                    niqe_ema_slow = bs_n * niqe_val_raw + (1.0 - bs_n) * niqe_ema_slow
                C = float(max(0.0, (niqe_ema_fast - niqe_ema_slow) / (niqe_ema_slow + 1e-9)))

            if needs_multi and mp_raw is not None:
                bf_mp, bs_mp = cfg.mp_fast_beta, cfg.mp_slow_beta
                proxy_weights = {
                    "L": cfg.mp_w_laplacian, "H": cfg.mp_w_entropy,
                    "tenengrad": cfg.mp_w_tenengrad, "edge_density": cfg.mp_w_edge_density,
                    "local_contrast": cfg.mp_w_local_contrast, "brenner": cfg.mp_w_brenner,
                    "color_entropy": cfg.mp_w_color_entropy,
                }
                weighted_drop_sum, w_sum = 0.0, 0.0
                for pname, pval in mp_raw.items():
                    w = proxy_weights.get(pname, 0.0)
                    if w < 1e-12:
                        continue
                    pval_f = float(pval)
                    if pname not in mp_ema:
                        mp_ema[pname] = (pval_f, pval_f)
                    else:
                        fast_prev, slow_prev = mp_ema[pname]
                        mp_ema[pname] = (bf_mp * pval_f + (1.0 - bf_mp) * fast_prev,
                                         bs_mp * pval_f + (1.0 - bs_mp) * slow_prev)
                    fast_v, slow_v = mp_ema[pname]
                    drop = abs(fast_v - slow_v) / (slow_v + 1e-9)
                    weighted_drop_sum += w * drop
                    w_sum += w
                C = float(weighted_drop_sum / w_sum) if w_sum > 1e-12 else 0.0
                rolling_C.add(C)

            avg_T_total = float(sum(latency_hist) / max(1, len(latency_hist))) if latency_hist else 0.0

            penalty = 0.0
            if cfg.mode != "fixed":
                deficit = max(0.0, avg_T_total - cfg.latency_budget_ms)
                penalty = deficit * cfg.latency_penalty_factor
                if avg_T_total > cfg.latency_budget_ms * (1.0 + cfg.budget_guard_margin):
                    budget_guard_left = max(budget_guard_left, cfg.budget_guard_frames)

            have_enough = len(rolling_C) >= max(10, cfg.history_window_size // 2)
            if cfg.mode == "fixed" or not have_enough:
                c_low, c_high, c_mid = cfg.c_low, cfg.c_high, cfg.combined_mid
            else:
                c_mid = float(np.clip(rolling_C.percentile(cfg.thr_mid) + penalty, 0.0, 1.0))
                c_low = float(np.clip(rolling_C.percentile(cfg.thr_lo) + penalty, 0.0, 1.0))
                c_high = float(np.clip(rolling_C.percentile(cfg.thr_hi) + penalty, 0.0, 1.0))
                if c_low > c_high:
                    c_low, c_high = c_high, c_low

            # Policy decision
            if cfg.policy == "n_only":
                choice = "n"
            elif cfg.policy == "s_only":
                choice = "s"
            elif cfg.policy == "entropy_only":
                choice = "s" if float(H or 0) >= Hmed else "n"
            elif cfg.policy == "combined":
                choice = "s" if float(C or 0) >= c_mid else "n"
            elif cfg.policy == "combined_hyst":
                if last_choice is None:
                    last_choice = "n"
                choice = (("s" if float(C or 0) >= c_high else "n") if last_choice == "n"
                          else ("n" if float(C or 0) <= c_low else "s"))
            elif cfg.policy == "conf_ema":
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                if kept < 5:
                    choice = "n"
                else:
                    choice = (("s" if c_val >= cfg.conf_ema_c_high else "n") if last_choice == "n"
                              else ("n" if c_val <= cfg.conf_ema_c_low else "s"))
            elif cfg.policy == "niqe_switch":
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                choice = (("s" if c_val >= cfg.niqe_c_high else "n") if last_choice == "n"
                          else ("n" if c_val <= cfg.niqe_c_low else "s"))
            elif cfg.policy == "multi_proxy":
                if last_choice is None:
                    last_choice = "n"
                c_val = float(C or 0)
                choice = (("s" if c_val >= cfg.mp_c_high else "n") if last_choice == "n"
                          else ("n" if c_val <= cfg.mp_c_low else "s"))
            else:
                choice = "n"

            # Budget guard
            if cfg.mode != "fixed" and cfg.policy in ("combined_hyst", "conf_ema", "niqe_switch", "multi_proxy"):
                if budget_guard_left > 0:
                    choice = "n"

            # Zero-detection gate
            zero_det_gated = False
            if (cfg.zero_det_gate and cfg.policy not in ("n_only", "s_only")
                    and choice == "s" and len(recent_det_counts) >= cfg.zero_det_lookback
                    and all(d == 0 for d in recent_det_counts)):
                choice = "n"
                zero_det_gated = True

            # Actuator protections
            if cfg.policy in SWITCHING_POLICIES:
                if last_choice is None:
                    last_choice = choice
                while switch_events and switch_events[0] <= kept - 100:
                    switch_events.popleft()
                requested_switch = choice != last_choice
                if requested_switch and dwell < cfg.min_dwell_frames:
                    choice = last_choice
                    requested_switch = False
                if requested_switch and len(switch_events) >= cfg.max_switches_per_100:
                    choice = last_choice
                    requested_switch = False
                if choice == last_choice:
                    dwell += 1
                else:
                    self._switches += 1
                    switch_events.append(kept)
                    dwell = 1
                    last_choice = choice
            else:
                dwell += 1

            T_ctrl_ms = (time.perf_counter() - t0_ctrl) * 1000.0

            # -- 3) INFERENCE --
            T_infer_n_ms = 0.0
            T_infer_s_ms = 0.0
            t0 = time.perf_counter()
            if choice == "n":
                dets = run_model_single(self.model_n, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)
                T_infer_n_ms = (time.perf_counter() - t0) * 1000.0
            else:
                dets = run_model_single(self.model_s, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)
                T_infer_s_ms = (time.perf_counter() - t0) * 1000.0
                self._slow_frames += 1

            T_total_ms = T_scene_ms + T_ctrl_ms + T_infer_n_ms + T_infer_s_ms
            latency_hist.append(T_total_ms)
            if budget_guard_left > 0:
                budget_guard_left -= 1

            cur_mean_conf = float(np.mean([float(d[4]) for d in dets])) if dets else 0.0
            if needs_conf_ema:
                last_mean_conf = cur_mean_conf

            if cfg.zero_det_gate:
                n_above = sum(1 for d in dets if float(d[4]) >= cfg.zero_det_conf_thresh)
                recent_det_counts.append(n_above)

            niqe_val_trace = float(niqe_val_raw) if needs_niqe else 0.0

            # -- 4) ANNOTATE & WRITE VIDEO --
            if self._video_writer is not None and self._video_writer.isOpened():
                header = [
                    f"Policy={cfg.policy} | Model={'YOLOv8n' if choice == 'n' else 'YOLOv8s'} | dwell={dwell}",
                    f"T_total={T_total_ms:.1f}ms | avg={avg_T_total:.1f}ms",
                ]
                vis = draw_overlay(frame, dets, choice, header, inf.conf_show)
                self._video_writer.write(vis)

            # -- 5) EMIT --
            fr = FrameResult(
                frame_idx=kept, choice=choice, num_detections=len(dets),
                L=float(L) if L is not None else 0.0,
                H=float(H) if H is not None else 0.0,
                C=float(C) if C is not None else 0.0,
                c_low=float(c_low), c_high=float(c_high),
                penalty=float(penalty), avg_T_total=float(avg_T_total),
                T_scene_ms=T_scene_ms, T_ctrl_ms=T_ctrl_ms,
                T_infer_n_ms=T_infer_n_ms, T_infer_s_ms=T_infer_s_ms,
                T_total_ms=T_total_ms, dwell=dwell,
                budget_guard_left=budget_guard_left,
                mean_conf=cur_mean_conf, conf_drop=conf_drop_val,
                niqe_score=niqe_val_trace, zero_det_gated=zero_det_gated,
            )

            self._totals.append(T_total_ms)
            self._t_scene.append(T_scene_ms)
            self._t_ctrl.append(T_ctrl_ms)
            self._t_in_n.append(T_infer_n_ms)
            self._t_in_s.append(T_infer_s_ms)
            self._L_trace.append(float(L) if L is not None else 0.0)
            self._H_trace.append(float(H) if H is not None else 0.0)
            self._dwell_trace.append(dwell)
            self._det_trace.append(len(dets))
            self._penalty_trace.append(float(penalty))
            self._avg_total_trace.append(float(avg_T_total))
            self._frame_indices.append(kept)
            self._C_trace.append(float(C) if C is not None else 0.0)
            self._choice_trace.append(choice)
            self._c_low_trace.append(float(c_low))
            self._c_high_trace.append(float(c_high))
            self._mean_conf_trace.append(cur_mean_conf)
            self._conf_drop_trace.append(conf_drop_val)
            self._niqe_trace.append(niqe_val_trace)
            self._zero_det_gated_trace.append(zero_det_gated)

            yield fr

            kept += 1
            if inf.max_frames > 0 and kept >= inf.max_frames:
                break

        cap.release()
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._build_summary()

    def _build_summary(self):
        totals = np.asarray(self._totals, dtype=np.float32)
        n = len(totals)
        t_in_n_arr = np.asarray(self._t_in_n, dtype=np.float32)
        t_in_s_arr = np.asarray(self._t_in_s, dtype=np.float32)
        n_ran = t_in_n_arr > 0
        s_ran = t_in_s_arr > 0
        T_infer_n_cond = float(t_in_n_arr[n_ran].mean()) if n_ran.any() else 0.0
        T_infer_s_cond = float(t_in_s_arr[s_ran].mean()) if s_ran.any() else 0.0
        zdg_count = sum(1 for g in self._zero_det_gated_trace if g)
        out_size = 0.0
        if self._output_video_path and os.path.isfile(self._output_video_path):
            out_size = round(os.path.getsize(self._output_video_path) / (1024 * 1024), 2)

        self._summary = RunSummary(
            video=os.path.basename(self.source_path), run_name=self.cfg.name,
            policy=self.cfg.policy, mode=self.cfg.mode, total_frames=n,
            T_scene_ms_mean=float(np.mean(self._t_scene)) if n else 0.0,
            T_ctrl_ms_mean=float(np.mean(self._t_ctrl)) if n else 0.0,
            T_infer_n_ms_mean=float(np.mean(self._t_in_n)) if n else 0.0,
            T_infer_s_ms_mean=float(np.mean(self._t_in_s)) if n else 0.0,
            T_infer_n_ms_cond_mean=T_infer_n_cond,
            T_infer_s_ms_cond_mean=T_infer_s_cond,
            T_total_ms_mean=float(np.mean(totals)) if n else 0.0,
            T_total_ms_p95=float(np.quantile(totals, 0.95)) if n else 0.0,
            T_total_ms_p99=float(np.quantile(totals, 0.99)) if n else 0.0,
            slow_pct=100.0 * (self._slow_frames / max(1, n)),
            switches=self._switches,
            sw_per_100=100.0 * (self._switches / max(1, n)),
            zero_det_gate_activations=zdg_count,
            run_timestamp=datetime.datetime.now().isoformat(),
            output_video_path=self._output_video_path,
            output_video_size_mb=out_size,
            frame_indices=list(self._frame_indices),
            T_total_trace=list(self._totals),
            T_scene_trace=list(self._t_scene),
            T_ctrl_trace=list(self._t_ctrl),
            T_infer_n_trace=list(self._t_in_n),
            T_infer_s_trace=list(self._t_in_s),
            C_trace=list(self._C_trace),
            L_trace=list(self._L_trace),
            H_trace=list(self._H_trace),
            choice_trace=list(self._choice_trace),
            c_low_trace=list(self._c_low_trace),
            c_high_trace=list(self._c_high_trace),
            dwell_trace=list(self._dwell_trace),
            num_detections_trace=list(self._det_trace),
            penalty_trace=list(self._penalty_trace),
            avg_T_total_trace=list(self._avg_total_trace),
            mean_conf_trace=list(self._mean_conf_trace),
            conf_drop_trace=list(self._conf_drop_trace),
            niqe_trace=list(self._niqe_trace),
            zero_det_gated_trace=list(self._zero_det_gated_trace),
        )

    def get_summary(self) -> Optional[RunSummary]:
        return self._summary


# =====================================================================
# 5. DETECTION QUALITY ANALYSIS (dual-model, s_only as oracle)
# =====================================================================

class PolicySimulator:
    """Replays policy logic on precomputed proxy signals (no inference needed)."""

    def __init__(self, policy: str, cfg: RunConfig):
        self.policy = policy
        self.cfg = cfg
        self.last_choice: Optional[str] = None
        self.dwell = 0
        self.switches = 0
        self.switch_events: deque = deque()
        self.frame_count = 0

        # Rolling windows
        self.rolling_L = FastRollingPercentile(cfg.history_window_size)
        self.rolling_H = FastRollingPercentile(cfg.history_window_size)
        self.rolling_C = FastRollingPercentile(cfg.history_window_size)
        self.c_ema_state: Optional[float] = None

        # conf_ema
        self.conf_ema_fast: Optional[float] = None
        self.conf_ema_slow: Optional[float] = None

        # niqe_switch
        self.niqe_ema_fast: Optional[float] = None
        self.niqe_ema_slow: Optional[float] = None

        # multi_proxy
        self.mp_ema: dict = {}

    def decide(self, L: float, H: float, mean_conf_n: float = 0.0,
               niqe_val: float = 0.0, extended_proxies: Optional[dict] = None,
               color_entropy: float = 0.0, mean_conf_s: float = 0.0) -> str:
        cfg = self.cfg
        policy = self.policy
        C: Optional[float] = None

        # Scene proxy normalization
        if policy in ("entropy_only", "combined", "combined_hyst"):
            self.rolling_L.add(L)
            self.rolling_H.add(H)
            L_lo, L_hi = self.rolling_L.percentile(0), self.rolling_L.percentile(100)
            H_lo, H_hi = self.rolling_H.percentile(0), self.rolling_H.percentile(100)
            Ln = float(np.clip((L - L_lo) / (L_hi - L_lo + 1e-9), 0.0, 1.0))
            Hn = float(np.clip((H - H_lo) / (H_hi - H_lo + 1e-9), 0.0, 1.0))
            C_raw = float(cfg.alpha * Ln + (1.0 - cfg.alpha) * Hn)
            if cfg.c_ema_beta <= 0.0:
                C = C_raw
            else:
                if self.c_ema_state is None:
                    self.c_ema_state = C_raw
                else:
                    self.c_ema_state = (1.0 - cfg.c_ema_beta) * self.c_ema_state + cfg.c_ema_beta * C_raw
                C = float(self.c_ema_state)
            self.rolling_C.add(C)

        # conf_ema: use the CHOSEN model's confidence from previous frame
        if policy == "conf_ema":
            # In detection quality mode, we use n-model confidence as the signal
            # (since in streaming mode, we'd use whichever model actually ran)
            mc = mean_conf_n if self.last_choice in (None, "n") else mean_conf_s
            bf, bs = cfg.conf_ema_fast_beta, cfg.conf_ema_slow_beta
            if self.conf_ema_fast is None:
                self.conf_ema_fast = mc
                self.conf_ema_slow = mc
            else:
                self.conf_ema_fast = bf * mc + (1.0 - bf) * self.conf_ema_fast
                self.conf_ema_slow = bs * mc + (1.0 - bs) * self.conf_ema_slow
            C = float(max(0.0, (self.conf_ema_slow - self.conf_ema_fast) / (self.conf_ema_slow + 1e-9)))

        if policy == "niqe_switch":
            bf, bs = cfg.niqe_fast_beta, cfg.niqe_slow_beta
            if self.niqe_ema_fast is None:
                self.niqe_ema_fast = niqe_val
                self.niqe_ema_slow = niqe_val
            else:
                self.niqe_ema_fast = bf * niqe_val + (1.0 - bf) * self.niqe_ema_fast
                self.niqe_ema_slow = bs * niqe_val + (1.0 - bs) * self.niqe_ema_slow
            C = float(max(0.0, (self.niqe_ema_fast - self.niqe_ema_slow) / (self.niqe_ema_slow + 1e-9)))

        if policy == "multi_proxy" and extended_proxies is not None:
            bf_mp, bs_mp = cfg.mp_fast_beta, cfg.mp_slow_beta
            proxy_weights = {
                "L": cfg.mp_w_laplacian, "H": cfg.mp_w_entropy,
                "tenengrad": cfg.mp_w_tenengrad, "edge_density": cfg.mp_w_edge_density,
                "local_contrast": cfg.mp_w_local_contrast, "brenner": cfg.mp_w_brenner,
                "color_entropy": cfg.mp_w_color_entropy,
            }
            all_proxies = {"L": L, "H": H, **extended_proxies, "color_entropy": color_entropy}
            wd_sum, w_sum = 0.0, 0.0
            for pname, pval in all_proxies.items():
                w = proxy_weights.get(pname, 0.0)
                if w < 1e-12:
                    continue
                pval_f = float(pval)
                if pname not in self.mp_ema:
                    self.mp_ema[pname] = (pval_f, pval_f)
                else:
                    fp, sp = self.mp_ema[pname]
                    self.mp_ema[pname] = (bf_mp * pval_f + (1.0 - bf_mp) * fp,
                                          bs_mp * pval_f + (1.0 - bs_mp) * sp)
                fv, sv = self.mp_ema[pname]
                drop = abs(fv - sv) / (sv + 1e-9)
                wd_sum += w * drop
                w_sum += w
            C = float(wd_sum / w_sum) if w_sum > 1e-12 else 0.0
            self.rolling_C.add(C)

        # Thresholds
        c_low, c_high, c_mid = cfg.c_low, cfg.c_high, cfg.combined_mid
        Hmed = self.rolling_H.percentile(50)

        # Policy decision
        if policy == "n_only":
            choice = "n"
        elif policy == "s_only":
            choice = "s"
        elif policy == "entropy_only":
            choice = "s" if H >= Hmed else "n"
        elif policy == "combined":
            choice = "s" if float(C or 0) >= c_mid else "n"
        elif policy == "combined_hyst":
            if self.last_choice is None:
                self.last_choice = "n"
            choice = (("s" if float(C or 0) >= c_high else "n") if self.last_choice == "n"
                      else ("n" if float(C or 0) <= c_low else "s"))
        elif policy == "conf_ema":
            if self.last_choice is None:
                self.last_choice = "n"
            c_val = float(C or 0)
            if self.frame_count < 5:
                choice = "n"
            else:
                choice = (("s" if c_val >= cfg.conf_ema_c_high else "n") if self.last_choice == "n"
                          else ("n" if c_val <= cfg.conf_ema_c_low else "s"))
        elif policy == "niqe_switch":
            if self.last_choice is None:
                self.last_choice = "n"
            c_val = float(C or 0)
            choice = (("s" if c_val >= cfg.niqe_c_high else "n") if self.last_choice == "n"
                      else ("n" if c_val <= cfg.niqe_c_low else "s"))
        elif policy == "multi_proxy":
            if self.last_choice is None:
                self.last_choice = "n"
            c_val = float(C or 0)
            choice = (("s" if c_val >= cfg.mp_c_high else "n") if self.last_choice == "n"
                      else ("n" if c_val <= cfg.mp_c_low else "s"))
        else:
            choice = "n"

        # Actuator protections
        if policy in SWITCHING_POLICIES:
            if self.last_choice is None:
                self.last_choice = choice
            while self.switch_events and self.switch_events[0] <= self.frame_count - 100:
                self.switch_events.popleft()
            requested_switch = choice != self.last_choice
            if requested_switch and self.dwell < cfg.min_dwell_frames:
                choice = self.last_choice
                requested_switch = False
            if requested_switch and len(self.switch_events) >= cfg.max_switches_per_100:
                choice = self.last_choice
                requested_switch = False
            if choice == self.last_choice:
                self.dwell += 1
            else:
                self.switches += 1
                self.switch_events.append(self.frame_count)
                self.dwell = 1
                self.last_choice = choice
        else:
            self.dwell += 1

        self.frame_count += 1
        return choice


def run_detection_quality(video_path: str, material: str,
                          model_n, model_s, inf: InferenceParams,
                          out_dir: Path) -> Dict[str, Any]:
    """Run BOTH models on every frame, simulate all 8 policies, compute detection coverage."""
    print(f"\n{'='*60}")
    print(f"Detection Quality: {os.path.basename(video_path)} ({material})")
    print(f"{'='*60}")

    cfg = RunConfig()
    sims = {p: PolicySimulator(p, cfg) for p in POLICIES}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_data = []  # per-frame records
    policy_choices = {p: [] for p in POLICIES}

    kept = 0
    raw_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if inf.stride > 1 and (raw_idx % inf.stride) != 0:
            raw_idx += 1
            continue
        raw_idx += 1

        # Run both models
        dets_n = run_model_single(model_n, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)
        dets_s = run_model_single(model_s, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)

        # Compute all proxies
        L, H = complexity_proxies_fast(frame, cfg.proxy_size, cfg.hist_bins)
        niqe_val = compute_nriqa_score(frame, cfg.proxy_size)

        small = cv2.resize(frame, (cfg.proxy_size, cfg.proxy_size), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        ext = compute_extended_proxies(gray_small)
        col_ent = compute_color_entropy(small)

        mean_conf_n = float(np.mean([float(d[4]) for d in dets_n])) if dets_n else 0.0
        mean_conf_s = float(np.mean([float(d[4]) for d in dets_s])) if dets_s else 0.0

        # Simulate all policies
        for p in POLICIES:
            choice = sims[p].decide(L, H, mean_conf_n, niqe_val, ext, col_ent, mean_conf_s)
            policy_choices[p].append(choice)

        frame_data.append({
            "frame_idx": kept, "n_dets": len(dets_n), "s_dets": len(dets_s),
            "dets_n": dets_n, "dets_s": dets_s,
            "mean_conf_n": mean_conf_n, "mean_conf_s": mean_conf_s,
            "L": L, "H": H, "niqe": niqe_val,
        })

        kept += 1
        if kept % 500 == 0:
            print(f"  Frame {kept}/{total_frames}...")
        if inf.max_frames > 0 and kept >= inf.max_frames:
            break

    cap.release()

    # Compute detection coverage per policy (s_only as oracle)
    results = {}
    for p in POLICIES:
        matched = 0
        missed = 0
        false_pos = 0
        oracle_total = 0
        conf_weighted_matched = 0.0
        conf_weighted_oracle = 0.0

        for i, fd in enumerate(frame_data):
            oracle_dets = fd["dets_s"]
            choice = policy_choices[p][i]
            policy_dets = fd["dets_n"] if choice == "n" else fd["dets_s"]

            # Filter by conf_show
            oracle_high = [d for d in oracle_dets if float(d[4]) >= inf.conf_show]
            policy_high = [d for d in policy_dets if float(d[4]) >= inf.conf_show]

            oracle_total += len(oracle_high)
            for od in oracle_high:
                conf_weighted_oracle += float(od[4])

            # Match policy detections to oracle via IoU
            used_policy = set()
            for od in oracle_high:
                best_iou = 0.0
                best_j = -1
                for j, pd in enumerate(policy_high):
                    if j in used_policy:
                        continue
                    iou = box_iou_xyxy(od[:4], pd[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= 0.5 and best_j >= 0:
                    matched += 1
                    conf_weighted_matched += float(od[4])
                    used_policy.add(best_j)
                else:
                    missed += 1
            false_pos += max(0, len(policy_high) - len(used_policy))

        precision = matched / max(1, matched + false_pos)
        recall = matched / max(1, oracle_total)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        cw_cov = conf_weighted_matched / max(1e-9, conf_weighted_oracle)

        s_pct = sum(1 for c in policy_choices[p] if c == "s") / max(1, len(policy_choices[p])) * 100
        sw = sims[p].switches
        sw100 = sw / max(1, len(policy_choices[p])) * 100

        results[p] = {
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "conf_weighted_coverage": round(cw_cov * 100, 1),
            "s_usage_pct": round(s_pct, 1),
            "switches": sw,
            "sw_per_100": round(sw100, 2),
            "total_frames": kept,
        }
        print(f"  {p:20s} | P={precision*100:5.1f}% R={recall*100:5.1f}% F1={f1*100:5.1f}% | s%={s_pct:5.1f}% | sw/100={sw100:.2f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "detection_quality.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =====================================================================
# 6. REPEATED TRIALS (timing statistics)
# =====================================================================

def run_timing_trial(video_path: str, material: str, policy: str,
                     model_n, model_s, inf: InferenceParams) -> RunSummary:
    """Run a single policy on a single video via StreamingEngine."""
    cfg = RunConfig(name=f"{policy}_{material}", policy=policy)
    engine = StreamingEngine(video_path, cfg, model_n, model_s, inf)
    for _ in engine.run():
        pass
    return engine.get_summary()


def run_repeated_trials(video_path: str, material: str,
                        model_n, model_s, inf: InferenceParams,
                        out_dir: Path, n_trials: int = 3) -> Dict[str, Any]:
    """Run all 8 policies N times each, collect timing stats."""
    print(f"\n{'='*60}")
    print(f"Repeated Trials ({n_trials}x): {os.path.basename(video_path)}")
    print(f"{'='*60}")

    results = {}
    for p in POLICIES:
        trial_means = []
        trial_p95s = []
        trial_sws = []
        for t in range(n_trials):
            summary = run_timing_trial(video_path, material, p, model_n, model_s, inf)
            trial_means.append(summary.T_total_ms_mean)
            trial_p95s.append(summary.T_total_ms_p95)
            trial_sws.append(summary.sw_per_100)
            print(f"  {p:20s} trial {t+1}/{n_trials}: mean={summary.T_total_ms_mean:.1f}ms p95={summary.T_total_ms_p95:.1f}ms")

        results[p] = {
            "mean_of_means": round(float(np.mean(trial_means)), 1),
            "std_of_means": round(float(np.std(trial_means)), 1),
            "mean_p95": round(float(np.mean(trial_p95s)), 1),
            "mean_sw_per_100": round(float(np.mean(trial_sws)), 2),
            "trial_means": [round(x, 1) for x in trial_means],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "repeated_trials.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# =====================================================================
# 7. FRAME-LEVEL VALIDATION (30 samples/policy, side-by-side images)
# =====================================================================

def run_frame_validation(video_path: str, material: str,
                         model_n, model_s, inf: InferenceParams,
                         out_dir: Path, n_samples: int = 30) -> Dict[str, Any]:
    """Sample 30 n-routed + 30 s-routed frames per switching policy.
    Run BOTH models, create side-by-side annotated images, compute IoU."""
    print(f"\n{'='*60}")
    print(f"Frame-Level Validation ({n_samples} samples/route): {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # First pass: run streaming engine for each switching policy to get choices
    policy_frame_choices = {}
    for p in SWITCHING_POLICIES:
        cfg = RunConfig(name=f"val_{p}", policy=p)
        engine = StreamingEngine(video_path, cfg, model_n, model_s, inf)
        choices = []
        for fr in engine.run():
            choices.append((fr.frame_idx, fr.choice, fr.C))
        policy_frame_choices[p] = choices
        print(f"  {p}: {len(choices)} frames, {sum(1 for _, c, _ in choices if c == 's')} s-routed")

    # Second pass: for each policy, sample n_samples frames per route
    results = {}
    for p in SWITCHING_POLICIES:
        choices = policy_frame_choices[p]
        n_frames = [(idx, c_val) for idx, ch, c_val in choices if ch == "n"]
        s_frames = [(idx, c_val) for idx, ch, c_val in choices if ch == "s"]

        # Sample
        rng = np.random.RandomState(42)
        n_sample = sorted(rng.choice(len(n_frames), min(n_samples, len(n_frames)), replace=False)) if n_frames else []
        s_sample = sorted(rng.choice(len(s_frames), min(n_samples, len(s_frames)), replace=False)) if s_frames else []

        n_sample_indices = {n_frames[i][0]: n_frames[i][1] for i in n_sample}
        s_sample_indices = {s_frames[i][0]: s_frames[i][1] for i in s_sample}
        all_needed = set(n_sample_indices.keys()) | set(s_sample_indices.keys())

        if not all_needed:
            results[p] = {"n_samples": 0, "s_samples": 0}
            continue

        # Read those frames and run both models
        p_dir = out_dir / p
        (p_dir / "n_routed").mkdir(parents=True, exist_ok=True)
        (p_dir / "s_routed").mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_results = []
        raw_idx = 0
        kept = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if inf.stride > 1 and (raw_idx % inf.stride) != 0:
                raw_idx += 1
                continue
            raw_idx += 1

            if kept in all_needed:
                dets_n = run_model_single(model_n, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)
                dets_s = run_model_single(model_s, frame, inf.imgsz, inf.device, inf.conf_min, inf.iou_nms)

                # Compute IoU between detection sets
                ious = []
                for dn in dets_n:
                    if float(dn[4]) < inf.conf_show:
                        continue
                    best_iou = 0.0
                    for ds in dets_s:
                        if float(ds[4]) < inf.conf_show:
                            continue
                        iou = box_iou_xyxy(dn[:4], ds[:4])
                        best_iou = max(best_iou, iou)
                    ious.append(best_iou)
                mean_iou = float(np.mean(ious)) if ious else 0.0

                route = "n" if kept in n_sample_indices else "s"
                c_val = n_sample_indices.get(kept, s_sample_indices.get(kept, 0.0))

                # Create side-by-side image
                vis_n = draw_overlay(frame, dets_n, "n", [f"YOLOv8n | {len(dets_n)} dets"], inf.conf_show)
                vis_s = draw_overlay(frame, dets_s, "s", [f"YOLOv8s | {len(dets_s)} dets"], inf.conf_show)
                side_by_side = np.hstack([vis_n, vis_s])

                route_dir = "n_routed" if route == "n" else "s_routed"
                fname = f"frame_{kept:06d}_iou{mean_iou:.3f}.png"
                cv2.imwrite(str(p_dir / route_dir / fname), side_by_side)

                frame_results.append({
                    "frame_idx": kept, "route": route, "c_val": round(c_val, 4),
                    "n_dets": len([d for d in dets_n if float(d[4]) >= inf.conf_show]),
                    "s_dets": len([d for d in dets_s if float(d[4]) >= inf.conf_show]),
                    "mean_iou": round(mean_iou, 4),
                })

            kept += 1
            if inf.max_frames > 0 and kept >= inf.max_frames:
                break
        cap.release()

        # Create grids (5x6 thumbnails)
        for route in ["n_routed", "s_routed"]:
            imgs_dir = p_dir / route
            img_files = sorted(imgs_dir.glob("*.png"))[:30]
            if img_files:
                _create_image_grid(img_files, p_dir / f"grid_{route}.png", cols=6, rows=5,
                                   title=f"{p} - {route.replace('_', ' ')} samples")

        # Stats
        n_results = [r for r in frame_results if r["route"] == "n"]
        s_results = [r for r in frame_results if r["route"] == "s"]
        n_mean_iou = float(np.mean([r["mean_iou"] for r in n_results])) if n_results else 0.0
        s_mean_iou = float(np.mean([r["mean_iou"] for r in s_results])) if s_results else 0.0

        results[p] = {
            "n_samples": len(n_results), "s_samples": len(s_results),
            "n_mean_iou": round(n_mean_iou, 4), "s_mean_iou": round(s_mean_iou, 4),
            "iou_gap": round(s_mean_iou - n_mean_iou, 4) if s_results else 0.0,
            "n_avg_dets": round(float(np.mean([r["n_dets"] for r in n_results])), 1) if n_results else 0,
            "s_avg_dets": round(float(np.mean([r["s_dets"] for r in s_results])), 1) if s_results else 0,
        }

        # Save per-policy CSV
        with open(p_dir / "validation_stats.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "route", "c_val", "n_dets", "s_dets", "mean_iou"])
            writer.writeheader()
            writer.writerows(frame_results)

        print(f"  {p:20s} | n_IoU={n_mean_iou:.3f} s_IoU={s_mean_iou:.3f} gap={results[p]['iou_gap']:+.3f}")

    # Save grand summary
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "validation_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def _create_image_grid(img_paths: list, out_path: Path, cols: int = 6, rows: int = 5,
                       title: str = "", thumb_w: int = 320, thumb_h: int = 180):
    """Create a grid of thumbnail images."""
    grid = np.zeros((rows * thumb_h + 40, cols * thumb_w, 3), dtype=np.uint8)
    if title:
        cv2.putText(grid, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, p in enumerate(img_paths[:rows * cols]):
        r, c = i // cols, i % cols
        img = cv2.imread(str(p))
        if img is not None:
            thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            y = r * thumb_h + 40
            grid[y:y + thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb
    cv2.imwrite(str(out_path), grid)


# =====================================================================
# 8. OVERNIGHT PIPELINE (all policies x all videos)
# =====================================================================

def run_overnight_pipeline(videos: Dict[str, List[Tuple[str, str]]],
                           model_paths: Dict[str, Tuple[str, str]],
                           inf: InferenceParams, out_dir: Path,
                           annotate_policies: Optional[List[str]] = None,
                           zero_det_gate: bool = False):
    """Run all 8 policies on all videos. Full overnight pipeline.

    Args:
        videos: {material: [(video_path, short_name), ...]}
        model_paths: {material: (n_model_path, s_model_path)}
        inf: InferenceParams
        out_dir: output root directory
        annotate_policies: which policies get annotated video output
        zero_det_gate: enable zero-detection gate
    """
    if annotate_policies is None:
        annotate_policies = []

    grand_results = {}

    for material, video_list in videos.items():
        n_path, s_path = model_paths[material]
        print(f"\nLoading {material} models...")
        model_n = YOLO(n_path)
        model_s = YOLO(s_path)

        for video_path, short_name in video_list:
            vid_dir = out_dir / material / short_name
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_results = {}

            for policy in POLICIES:
                print(f"\n--- {material}/{short_name} | {policy} ---")
                cfg = RunConfig(
                    name=f"{policy}_{short_name}",
                    policy=policy,
                    zero_det_gate=zero_det_gate,
                )

                # Save annotated video for selected policies
                if policy in annotate_policies:
                    cfg.save_annotated_video = True
                    cfg.output_video_path = str(vid_dir / f"{policy}_annotated.mp4")

                engine = StreamingEngine(video_path, cfg, model_n, model_s, inf)
                for fr in engine.run():
                    pass
                summary = engine.get_summary()

                if summary:
                    # Save per-policy summary
                    with open(vid_dir / f"{policy}_summary.json", "w") as f:
                        json.dump(summary.to_dict(), f, indent=2)

                    # Save per-frame trace CSV
                    with open(vid_dir / f"{policy}_trace.csv", "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["frame_idx", "choice", "T_total_ms", "T_scene_ms",
                                         "T_ctrl_ms", "T_infer_n_ms", "T_infer_s_ms",
                                         "C", "L", "H", "dwell", "num_dets", "mean_conf"])
                        for j in range(summary.total_frames):
                            writer.writerow([
                                summary.frame_indices[j], summary.choice_trace[j],
                                f"{summary.T_total_trace[j]:.2f}", f"{summary.T_scene_trace[j]:.2f}",
                                f"{summary.T_ctrl_trace[j]:.2f}", f"{summary.T_infer_n_trace[j]:.2f}",
                                f"{summary.T_infer_s_trace[j]:.2f}", f"{summary.C_trace[j]:.4f}",
                                f"{summary.L_trace[j]:.2f}", f"{summary.H_trace[j]:.4f}",
                                summary.dwell_trace[j], summary.num_detections_trace[j],
                                f"{summary.mean_conf_trace[j]:.4f}",
                            ])

                    vid_results[policy] = {
                        "T_total_mean": summary.T_total_ms_mean,
                        "T_total_p95": summary.T_total_ms_p95,
                        "T_scene_mean": summary.T_scene_ms_mean,
                        "slow_pct": summary.slow_pct,
                        "sw_per_100": summary.sw_per_100,
                        "total_frames": summary.total_frames,
                    }

                    print(f"  T_total={summary.T_total_ms_mean:.1f}ms  slow%={summary.slow_pct:.1f}%  sw/100={summary.sw_per_100:.2f}")

            # Save cross-policy comparison for this video
            with open(vid_dir / "policy_comparison.json", "w") as f:
                json.dump(vid_results, f, indent=2)
            grand_results[f"{material}/{short_name}"] = vid_results

    # Save grand summary
    with open(out_dir / "grand_summary.json", "w") as f:
        json.dump(grand_results, f, indent=2)

    print(f"\nOvernight pipeline complete. Results in: {out_dir}")
    return grand_results


# =====================================================================
# 8. PLOT GENERATION (50+ charts)
# =====================================================================

def generate_all_plots(results_dir: Path):
    """Generate all publication-quality plots from experiment results."""
    if not HAS_MPL:
        print("ERROR: matplotlib not available. Install with: pip install matplotlib")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300,
                         "figure.figsize": (12, 7)})

    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    grand_json = results_dir / "grand_summary.json"
    det_quality_files = list(results_dir.rglob("detection_quality.json"))
    trial_files = list(results_dir.rglob("repeated_trials.json"))
    validation_files = list(results_dir.rglob("validation_summary.json"))

    # --- PLOT 1: Detection Quality Bar Chart ---
    for dq_file in det_quality_files:
        with open(dq_file) as f:
            dq = json.load(f)
        video_name = dq_file.parent.name
        _plot_detection_quality_bars(dq, figs_dir / f"detection_quality_{video_name}.png", video_name)

    # --- PLOT 2: Timing Comparison ---
    for tf in trial_files:
        with open(tf) as f:
            trials = json.load(f)
        video_name = tf.parent.name
        _plot_timing_comparison(trials, figs_dir / f"timing_comparison_{video_name}.png", video_name)

    # --- PLOT 3: Validation IoU ---
    for vf in validation_files:
        with open(vf) as f:
            val = json.load(f)
        video_name = vf.parent.name
        _plot_validation_iou(val, figs_dir / f"validation_iou_{video_name}.png", video_name)

    # --- PLOTS from grand summary ---
    if grand_json.exists():
        with open(grand_json) as f:
            grand = json.load(f)
        _plot_grand_timing_heatmap(grand, figs_dir / "grand_timing_heatmap.png")
        _plot_grand_pareto(grand, figs_dir / "grand_pareto_front.png")
        _plot_grand_slow_pct(grand, figs_dir / "grand_slow_pct.png")
        _plot_grand_switching_rate(grand, figs_dir / "grand_switching_rate.png")
        _plot_grand_speedup_vs_sonly(grand, figs_dir / "grand_speedup_vs_sonly.png")

    # --- Per-video trace plots ---
    summary_files = list(results_dir.rglob("*_summary.json"))
    for sf in summary_files:
        if sf.name == "grand_summary.json" or sf.name == "policy_comparison.json":
            continue
        try:
            with open(sf) as f:
                sd = json.load(f)
            s = RunSummary.from_dict(sd)
            if s.total_frames > 0:
                policy = s.policy
                video = s.video.replace(".mp4", "")
                _plot_timing_timeseries(s, figs_dir / f"timeseries_{video}_{policy}.png")
                if policy in SWITCHING_POLICIES:
                    _plot_choice_timeline(s, figs_dir / f"choice_{video}_{policy}.png")
        except Exception:
            continue

    # --- Cross-policy comparison per video ---
    comparison_files = list(results_dir.rglob("policy_comparison.json"))
    for cf in comparison_files:
        with open(cf) as f:
            comp = json.load(f)
        video_name = cf.parent.name
        _plot_policy_comparison_bars(comp, figs_dir / f"comparison_{video_name}.png", video_name)

    print(f"\nAll plots saved to: {figs_dir}")
    print(f"Total figures: {len(list(figs_dir.glob('*.png')))}")


# --- Individual plot functions ---

def _plot_detection_quality_bars(dq: dict, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    policies = [p for p in POLICIES if p in dq]
    colors = [POLICY_COLORS.get(p, "#999") for p in policies]

    for ax, metric, label in zip(axes, ["precision", "recall", "f1"],
                                  ["Precision (%)", "Recall / Coverage (%)", "F1 Score (%)"]):
        vals = [dq[p][metric] for p in policies]
        bars = ax.bar(range(len(policies)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"Detection Quality: {title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_timing_comparison(trials: dict, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    policies = [p for p in POLICIES if p in trials]
    means = [trials[p]["mean_of_means"] for p in policies]
    stds = [trials[p]["std_of_means"] for p in policies]
    colors = [POLICY_COLORS.get(p, "#999") for p in policies]

    bars = ax.bar(range(len(policies)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Mean T_total (ms)")
    ax.set_title(f"Timing Comparison: {title}")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 2, f"{m:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_validation_iou(val: dict, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    policies = [p for p in SWITCHING_POLICIES if p in val and val[p].get("n_samples", 0) > 0]
    if not policies:
        plt.close(fig)
        return

    # IoU comparison
    ax = axes[0]
    x = np.arange(len(policies))
    w = 0.35
    n_ious = [val[p]["n_mean_iou"] for p in policies]
    s_ious = [val[p]["s_mean_iou"] for p in policies]
    ax.bar(x - w / 2, n_ious, w, label="n-routed", color="#4CAF50", alpha=0.8)
    ax.bar(x + w / 2, s_ious, w, label="s-routed", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Mean IoU (n vs s detections)")
    ax.set_title("IoU by Routing Decision")
    ax.legend()

    # IoU gap
    ax = axes[1]
    gaps = [val[p]["iou_gap"] for p in policies]
    colors = ["#4CAF50" if g > 0 else "#F44336" for g in gaps]
    ax.bar(range(len(policies)), gaps, color=colors, edgecolor="white")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("IoU Gap (s-routed - n-routed)")
    ax.set_title("Routing Quality: Positive = s-routed frames ARE harder")
    ax.axhline(y=0, color="black", linewidth=0.8)

    fig.suptitle(f"Frame-Level Validation: {title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_grand_timing_heatmap(grand: dict, out_path: Path):
    videos = sorted(grand.keys())
    policies = POLICIES
    data = np.full((len(videos), len(policies)), np.nan)
    for i, v in enumerate(videos):
        for j, p in enumerate(policies):
            if p in grand[v]:
                data[i, j] = grand[v][p]["T_total_mean"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(videos) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_yticks(range(len(videos)))
    ax.set_yticklabels([v.split("/")[-1] for v in videos], fontsize=9)
    for i in range(len(videos)):
        for j in range(len(policies)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > np.nanmedian(data) else "black")
    plt.colorbar(im, ax=ax, label="Mean T_total (ms)")
    ax.set_title("Timing Heatmap: All Videos x All Policies", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_grand_pareto(grand: dict, out_path: Path):
    """Pareto front: T_total vs detection coverage (if available)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Aggregate across videos
    policy_timing = {p: [] for p in POLICIES}
    for v in grand.values():
        for p in POLICIES:
            if p in v:
                policy_timing[p].append(v[p]["T_total_mean"])

    for p in POLICIES:
        if policy_timing[p]:
            mean_t = np.mean(policy_timing[p])
            slow_pcts = [grand[v][p]["slow_pct"] for v in grand if p in grand[v]]
            mean_slow = np.mean(slow_pcts)
            ax.scatter(mean_t, mean_slow, s=120, c=POLICY_COLORS.get(p, "#999"),
                       edgecolors="black", linewidths=1, zorder=5)
            ax.annotate(p, (mean_t, mean_slow), textcoords="offset points",
                        xytext=(8, 8), fontsize=9)

    ax.set_xlabel("Mean T_total (ms)")
    ax.set_ylabel("s-model Usage (%)")
    ax.set_title("Pareto Front: Latency vs s-model Usage", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_grand_slow_pct(grand: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    policies = POLICIES
    means = []
    for p in policies:
        vals = [grand[v][p]["slow_pct"] for v in grand if p in grand[v]]
        means.append(np.mean(vals) if vals else 0)
    bars = ax.bar(range(len(policies)), means, color=[POLICY_COLORS.get(p, "#999") for p in policies])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("s-model Usage (%)")
    ax.set_title("s-model Usage Across All Videos", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1, f"{m:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_grand_switching_rate(grand: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    policies = SWITCHING_POLICIES
    means = []
    for p in policies:
        vals = [grand[v][p]["sw_per_100"] for v in grand if p in grand[v]]
        means.append(np.mean(vals) if vals else 0)
    bars = ax.bar(range(len(policies)), means, color=[POLICY_COLORS.get(p, "#999") for p in policies])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Switches per 100 frames")
    ax.set_title("Switching Rate Across All Videos", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.1, f"{m:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_grand_speedup_vs_sonly(grand: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    policies = [p for p in POLICIES if p != "s_only"]

    speedups = []
    for p in policies:
        ratios = []
        for v in grand:
            if p in grand[v] and "s_only" in grand[v]:
                s_time = grand[v]["s_only"]["T_total_mean"]
                p_time = grand[v][p]["T_total_mean"]
                if s_time > 0:
                    ratios.append(1.0 - p_time / s_time)
        speedups.append(np.mean(ratios) * 100 if ratios else 0)

    colors = [POLICY_COLORS.get(p, "#999") for p in policies]
    bars = ax.bar(range(len(policies)), speedups, color=colors)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Speedup vs s_only (%)")
    ax.set_title("Speedup Relative to s_only Baseline", fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 1, f"{s:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_timing_timeseries(s: RunSummary, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 5))
    frames = s.frame_indices
    totals = s.T_total_trace
    choices = s.choice_trace

    n_mask = [i for i, c in enumerate(choices) if c == "n"]
    s_mask = [i for i, c in enumerate(choices) if c == "s"]

    if n_mask:
        ax.scatter([frames[i] for i in n_mask], [totals[i] for i in n_mask],
                   s=2, c="#4CAF50", alpha=0.5, label="n-model")
    if s_mask:
        ax.scatter([frames[i] for i in s_mask], [totals[i] for i in s_mask],
                   s=2, c="#F44336", alpha=0.5, label="s-model")

    # Rolling average
    if len(totals) > 50:
        kernel = np.ones(50) / 50
        rolling = np.convolve(totals, kernel, mode="valid")
        ax.plot(frames[:len(rolling)], rolling, color="black", linewidth=1.5, alpha=0.7, label="50-frame avg")

    ax.set_xlabel("Frame")
    ax.set_ylabel("T_total (ms)")
    ax.set_title(f"Timing: {s.video} | {s.policy} | mean={s.T_total_ms_mean:.1f}ms")
    ax.legend(markerscale=5)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_choice_timeline(s: RunSummary, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 3, 2]})

    # Choice strip
    ax = axes[0]
    choice_binary = [1 if c == "s" else 0 for c in s.choice_trace]
    ax.imshow([choice_binary], aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
              extent=[0, len(choice_binary), 0, 1])
    ax.set_yticks([])
    ax.set_title(f"Model Choice: {s.video} | {s.policy}")
    ax.set_ylabel("n/s")

    # C-score
    ax = axes[1]
    ax.plot(s.frame_indices, s.C_trace, linewidth=0.5, color="#9C27B0", alpha=0.7)
    if s.c_low_trace and s.c_high_trace:
        ax.axhline(y=np.mean(s.c_low_trace), color="green", linestyle="--", alpha=0.5, label="c_low")
        ax.axhline(y=np.mean(s.c_high_trace), color="red", linestyle="--", alpha=0.5, label="c_high")
    ax.set_ylabel("C-score / Signal")
    ax.legend(fontsize=8)

    # Dwell
    ax = axes[2]
    ax.fill_between(s.frame_indices, s.dwell_trace, alpha=0.5, color="#FF9800")
    ax.set_ylabel("Dwell")
    ax.set_xlabel("Frame")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_policy_comparison_bars(comp: dict, out_path: Path, title: str):
    policies = [p for p in POLICIES if p in comp]
    if not policies:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Timing
    ax = axes[0]
    vals = [comp[p]["T_total_mean"] for p in policies]
    ax.bar(range(len(policies)), vals, color=[POLICY_COLORS.get(p, "#999") for p in policies])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Mean T_total (ms)")
    ax.set_title("Latency")

    # s-model usage
    ax = axes[1]
    vals = [comp[p]["slow_pct"] for p in policies]
    ax.bar(range(len(policies)), vals, color=[POLICY_COLORS.get(p, "#999") for p in policies])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("s-model Usage (%)")
    ax.set_title("s-model Usage")

    # Switching rate
    ax = axes[2]
    vals = [comp[p]["sw_per_100"] for p in policies]
    ax.bar(range(len(policies)), vals, color=[POLICY_COLORS.get(p, "#999") for p in policies])
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Switches/100")
    ax.set_title("Switching Rate")

    fig.suptitle(f"Policy Comparison: {title}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# =====================================================================
# 9. THESIS REPORT GENERATOR
# =====================================================================

def generate_report(results_dir: Path):
    """Generate a complete thesis analysis report (Markdown)."""
    report_path = results_dir / "THESIS_ANALYSIS_REPORT.md"

    lines = [
        "# DMS-Raptor: Complete Experiment Analysis Report",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Executive Summary",
        "",
        "This report presents the complete experimental analysis of the Dynamic Model ",
        "Switching (DMS) framework for real-time UAV power-line insulator inspection.",
        "The framework adaptively selects between YOLOv8n (nano, fast) and YOLOv8s ",
        "(small, accurate) on a per-frame basis using scene complexity proxies.",
        "",
    ]

    # Load grand summary
    grand_path = results_dir / "grand_summary.json"
    if grand_path.exists():
        with open(grand_path) as f:
            grand = json.load(f)

        lines += [
            "## 2. Experiment Scope",
            "",
            f"- **Videos tested:** {len(grand)}",
            f"- **Policies evaluated:** {len(POLICIES)} ({', '.join(POLICIES)})",
            "",
        ]

        # Aggregate timing table
        lines += [
            "## 3. Aggregate Timing Results",
            "",
            "| Policy | Mean T_total (ms) | s-Usage (%) | Switches/100 | Speedup vs s_only |",
            "|--------|------------------|-------------|--------------|-------------------|",
        ]

        for p in POLICIES:
            t_vals = [grand[v][p]["T_total_mean"] for v in grand if p in grand[v]]
            s_vals = [grand[v][p]["slow_pct"] for v in grand if p in grand[v]]
            sw_vals = [grand[v][p]["sw_per_100"] for v in grand if p in grand[v]]
            mean_t = np.mean(t_vals) if t_vals else 0
            mean_s = np.mean(s_vals) if s_vals else 0
            mean_sw = np.mean(sw_vals) if sw_vals else 0

            s_only_t = np.mean([grand[v]["s_only"]["T_total_mean"] for v in grand if "s_only" in grand[v]])
            speedup = (1.0 - mean_t / s_only_t) * 100 if s_only_t > 0 else 0

            lines.append(f"| {p} | {mean_t:.1f} | {mean_s:.1f} | {mean_sw:.2f} | {speedup:+.1f}% |")

        lines += [
            "",
            "## 4. Key Research Findings",
            "",
            "### Finding 1: C-Score Normalization Failure",
            "Rolling percentile normalization destroys proxy-target correlation.",
            "Raw Laplacian correlates moderately with model disagreement, but after ",
            "normalization the C-score correlation drops to near zero.",
            "",
            "### Finding 2: combined_hyst Conservative Strategy",
            "Despite weak proxy signal, combined_hyst achieves high detection coverage ",
            "by conservatively using the s-model for ~50% of frames. The hysteresis ",
            "mechanism provides stability with minimal switching.",
            "",
            "### Finding 3: conf_ema Best Speed-Accuracy Trade-off",
            "conf_ema achieves the best trade-off: zero image processing cost (T_scene=0), ",
            "best cross-video generalization, and competitive detection coverage with ",
            "only ~27% s-model usage.",
            "",
            "### Finding 4: GPU Deployment Boundary",
            "On GPU, n_only and s_only run at similar speeds (~18-22ms), eliminating the ",
            "latency gap that DMS exploits. DMS is designed for CPU/edge deployment.",
            "",
        ]

    # Detection quality
    dq_files = list(results_dir.rglob("detection_quality.json"))
    if dq_files:
        lines += [
            "## 5. Detection Quality (s_only as Oracle)",
            "",
            "| Policy | Precision (%) | Recall (%) | F1 (%) | s-Usage (%) |",
            "|--------|-------------|-----------|--------|------------|",
        ]
        # Use first file as representative
        with open(dq_files[0]) as f:
            dq = json.load(f)
        for p in POLICIES:
            if p in dq:
                d = dq[p]
                lines.append(f"| {p} | {d['precision']} | {d['recall']} | {d['f1']} | {d['s_usage_pct']} |")
        lines.append("")

    # Validation
    val_files = list(results_dir.rglob("validation_summary.json"))
    if val_files:
        lines += [
            "## 6. Frame-Level Proxy Routing Validation",
            "",
            "| Policy | n-routed IoU | s-routed IoU | IoU Gap | Routing Quality |",
            "|--------|-------------|-------------|---------|----------------|",
        ]
        with open(val_files[0]) as f:
            val = json.load(f)
        for p in SWITCHING_POLICIES:
            if p in val and val[p].get("n_samples", 0) > 0:
                v = val[p]
                quality = "Good" if v["iou_gap"] > 0.05 else ("Weak" if v["iou_gap"] > 0 else "Poor")
                lines.append(f"| {p} | {v['n_mean_iou']:.3f} | {v['s_mean_iou']:.3f} | {v['iou_gap']:+.3f} | {quality} |")
        lines += [
            "",
            "**Interpretation:** Positive IoU gap means s-routed frames genuinely have ",
            "lower agreement between models (harder frames), indicating good routing.",
            "",
        ]

    lines += [
        "## 7. Methodology Notes",
        "",
        "### Timing Architecture",
        "```",
        "T_total = T_scene + T_ctrl + T_infer   (only ONE model runs per frame)",
        "```",
        "- T_scene: Image proxy computation (0ms for conf_ema, ~3ms for L+H, ~30ms for NIQE)",
        "- T_ctrl: Controller logic (~0.05-0.2ms)",
        "- T_infer: YOLO forward pass + internal NMS",
        "",
        "### Pseudo-Oracle Validation",
        "Since no ground-truth bounding box annotations exist, we use s_only ",
        "detections as the oracle. Detection coverage = fraction of s_only ",
        "detections matched (IoU >= 0.5) by the policy's chosen model.",
        "",
        "### Actuator Stabilizers",
        "All switching policies include:",
        "- min_dwell_frames = 10 (no switching within 10 frames)",
        "- max_switches_per_100 = 12 (rate limiter)",
        "- C-score EMA smoothing (beta = 0.25)",
        "",
        "## 8. File Structure",
        "",
        "```",
        "results/",
        "  grand_summary.json          -- All videos x all policies",
        "  figures/                     -- All generated plots",
        "  <material>/<video>/",
        "    <policy>_summary.json      -- Per-policy run summary",
        "    <policy>_trace.csv         -- Per-frame trace data",
        "    <policy>_annotated.mp4     -- Annotated video (if enabled)",
        "    policy_comparison.json     -- Cross-policy comparison",
        "  detection_quality/",
        "    detection_quality.json     -- Detection coverage analysis",
        "  timing_trials/",
        "    repeated_trials.json       -- Multi-trial timing stats",
        "  frame_validation/",
        "    <policy>/",
        "      n_routed/                -- 30 side-by-side images (n-routed frames)",
        "      s_routed/                -- 30 side-by-side images (s-routed frames)",
        "      grid_n_routed.png        -- 5x6 thumbnail grid",
        "      grid_s_routed.png        -- 5x6 thumbnail grid",
        "      validation_stats.csv     -- Per-frame IoU data",
        "    validation_summary.json    -- Routing quality summary",
        "```",
        "",
        "---",
        f"*Report generated by DMS-Raptor Experiment Pipeline*",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nThesis analysis report saved to: {report_path}")


# =====================================================================
# 10. VIDEO/MODEL DISCOVERY
# =====================================================================

def discover_videos(videos_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Auto-discover videos organized by material type."""
    videos: Dict[str, List[Tuple[str, str]]] = {}
    base = Path(videos_dir)

    if not base.exists():
        print(f"WARNING: Videos directory not found: {base}")
        print(f"Create it or use --videos-dir to point to your videos.")
        return videos

    for material_dir in sorted(base.iterdir()):
        if not material_dir.is_dir():
            continue
        material = material_dir.name.replace("_insulator_videos", "").replace("_videos", "")
        vid_list = []
        for vf in sorted(material_dir.glob("*.mp4")):
            short = vf.stem[:20]  # truncate long names
            vid_list.append((str(vf), short))
        if vid_list:
            videos[material] = vid_list

    # Also check for videos directly in the directory
    direct_vids = list(base.glob("*.mp4"))
    if direct_vids and not videos:
        videos["default"] = [(str(v), v.stem[:20]) for v in sorted(direct_vids)]

    return videos


def discover_models(models_dir: str) -> Dict[str, Tuple[str, str]]:
    """Auto-discover model pairs organized by material."""
    models: Dict[str, Tuple[str, str]] = {}
    base = Path(models_dir)

    # Pattern: <material>_y8n_fast.pt / <material>_y8s_accurate.pt
    n_files = sorted(base.glob("*_y8n_*.pt")) + sorted(base.glob("*_n_*.pt")) + sorted(base.glob("*n*.pt"))
    s_files = sorted(base.glob("*_y8s_*.pt")) + sorted(base.glob("*_s_*.pt")) + sorted(base.glob("*s*.pt"))

    # Try standard naming convention
    for nf in base.glob("*_y8n_*.pt"):
        material = nf.stem.split("_y8n")[0]
        sf = base / nf.name.replace("y8n_fast", "y8s_accurate")
        if sf.exists():
            models[material] = (str(nf), str(sf))

    # Fallback: if only 2 .pt files, assume they're n and s
    if not models:
        pts = sorted(base.glob("*.pt"))
        if len(pts) == 2:
            # Smaller file is likely n-model
            if pts[0].stat().st_size <= pts[1].stat().st_size:
                models["default"] = (str(pts[0]), str(pts[1]))
            else:
                models["default"] = (str(pts[1]), str(pts[0]))

    return models


# =====================================================================
# 11. CLI ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DMS-Raptor: Dynamic Model Switching Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (detection + timing + validation + plots + report)
  python dms_experiment.py run-all --models-dir models --videos-dir videos --device cpu

  # Quick test (500 frames)
  python dms_experiment.py run-all --models-dir models --videos-dir videos --max-frames 500

  # Only detection quality analysis
  python dms_experiment.py detection --models-dir models --videos-dir videos

  # Only generate plots from existing results
  python dms_experiment.py plots --results-dir results

  # Only generate thesis report
  python dms_experiment.py report --results-dir results
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Experiment to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models-dir", default="models", help="Directory with .pt model files")
    common.add_argument("--videos-dir", default="videos", help="Directory with video files")
    common.add_argument("--results-dir", default="results", help="Output directory")
    common.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1"],
                        help="Inference device")
    common.add_argument("--max-frames", type=int, default=0, help="Max frames per video (0=all)")
    common.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    common.add_argument("--imgsz", type=int, default=640, help="YOLO input size")

    # run-all
    p_all = sub.add_parser("run-all", parents=[common], help="Run complete pipeline")
    p_all.add_argument("--trials", type=int, default=3, help="Number of timing trials")
    p_all.add_argument("--samples", type=int, default=30, help="Validation samples per route")
    p_all.add_argument("--annotate-policies", nargs="*", default=["conf_ema", "combined_hyst"],
                       help="Policies to generate annotated videos for")
    p_all.add_argument("--zero-det-gate", action="store_true", help="Enable zero-detection gate")

    # detection
    p_det = sub.add_parser("detection", parents=[common], help="Detection quality analysis")

    # timing
    p_tim = sub.add_parser("timing", parents=[common], help="Repeated timing trials")
    p_tim.add_argument("--trials", type=int, default=3)

    # validation
    p_val = sub.add_parser("validation", parents=[common], help="Frame-level validation")
    p_val.add_argument("--samples", type=int, default=30)

    # overnight
    p_night = sub.add_parser("overnight", parents=[common], help="Full overnight pipeline")
    p_night.add_argument("--annotate-policies", nargs="*", default=["conf_ema", "combined_hyst"])
    p_night.add_argument("--zero-det-gate", action="store_true")

    # plots
    p_plot = sub.add_parser("plots", help="Generate plots from results")
    p_plot.add_argument("--results-dir", default="results")

    # report
    p_rep = sub.add_parser("report", help="Generate thesis analysis report")
    p_rep.add_argument("--results-dir", default="results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Plot-only and report-only modes
    if args.command == "plots":
        generate_all_plots(Path(args.results_dir))
        return
    if args.command == "report":
        generate_report(Path(args.results_dir))
        return

    # Check dependencies
    if not HAS_YOLO:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Discover models and videos
    models = discover_models(args.models_dir)
    videos = discover_videos(args.videos_dir)

    if not models:
        print(f"ERROR: No model pairs found in {args.models_dir}")
        print("Expected: <material>_y8n_fast.pt + <material>_y8s_accurate.pt")
        sys.exit(1)
    if not videos:
        print(f"ERROR: No videos found in {args.videos_dir}")
        sys.exit(1)

    print(f"Models found: {list(models.keys())}")
    print(f"Videos found: {sum(len(v) for v in videos.values())} across {list(videos.keys())}")

    inf = InferenceParams(
        imgsz=args.imgsz, device=args.device,
        max_frames=args.max_frames, stride=args.stride,
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Map materials to model pairs (handle mismatched keys)
    def get_model_pair(material: str):
        if material in models:
            return models[material]
        # Try composite or default fallback
        for key in models:
            if material.startswith(key) or key.startswith(material):
                return models[key]
        # Use first available
        return next(iter(models.values()))

    if args.command == "detection":
        for material, vid_list in videos.items():
            n_path, s_path = get_model_pair(material)
            model_n, model_s = YOLO(n_path), YOLO(s_path)
            for vp, short in vid_list:
                run_detection_quality(vp, material, model_n, model_s, inf,
                                      results_dir / "detection_quality" / short)

    elif args.command == "timing":
        for material, vid_list in videos.items():
            n_path, s_path = get_model_pair(material)
            model_n, model_s = YOLO(n_path), YOLO(s_path)
            for vp, short in vid_list:
                run_repeated_trials(vp, material, model_n, model_s, inf,
                                    results_dir / "timing_trials" / short,
                                    n_trials=args.trials)

    elif args.command == "validation":
        for material, vid_list in videos.items():
            n_path, s_path = get_model_pair(material)
            model_n, model_s = YOLO(n_path), YOLO(s_path)
            for vp, short in vid_list:
                run_frame_validation(vp, material, model_n, model_s, inf,
                                     results_dir / "frame_validation" / short,
                                     n_samples=args.samples)

    elif args.command == "overnight":
        model_map = {m: get_model_pair(m) for m in videos}
        run_overnight_pipeline(videos, model_map, inf, results_dir / "overnight",
                               annotate_policies=args.annotate_policies,
                               zero_det_gate=args.zero_det_gate)

    elif args.command == "run-all":
        print("\n" + "=" * 70)
        print("DMS-RAPTOR COMPLETE EXPERIMENT PIPELINE")
        print("=" * 70)
        t_start = time.time()

        for material, vid_list in videos.items():
            n_path, s_path = get_model_pair(material)
            model_n, model_s = YOLO(n_path), YOLO(s_path)

            for vp, short in vid_list:
                # Phase 1: Detection Quality
                print(f"\n>>> Phase 1: Detection Quality ({short})")
                run_detection_quality(vp, material, model_n, model_s, inf,
                                      results_dir / "detection_quality" / short)

                # Phase 2: Repeated Trials
                print(f"\n>>> Phase 2: Timing Trials ({short})")
                run_repeated_trials(vp, material, model_n, model_s, inf,
                                    results_dir / "timing_trials" / short,
                                    n_trials=args.trials)

                # Phase 3: Frame Validation
                print(f"\n>>> Phase 3: Frame Validation ({short})")
                run_frame_validation(vp, material, model_n, model_s, inf,
                                     results_dir / "frame_validation" / short,
                                     n_samples=args.samples)

        # Phase 4: Overnight Pipeline
        print(f"\n>>> Phase 4: Full Pipeline")
        model_map = {m: get_model_pair(m) for m in videos}
        run_overnight_pipeline(videos, model_map, inf, results_dir / "overnight",
                               annotate_policies=args.annotate_policies,
                               zero_det_gate=args.zero_det_gate)

        # Phase 5: Plots
        print(f"\n>>> Phase 5: Generating Plots")
        generate_all_plots(results_dir)

        # Phase 6: Report
        print(f"\n>>> Phase 6: Generating Report")
        generate_report(results_dir)

        elapsed = time.time() - t_start
        print(f"\n{'=' * 70}")
        print(f"COMPLETE. Total time: {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")
        print(f"Results in: {results_dir}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
