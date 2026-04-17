from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet


def _gray_entropy(gray: np.ndarray, bins: int = 64) -> float:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist[hist > 1e-12] / total
    return float(-(p * np.log2(p)).sum())


def _color_entropy(img_bgr: np.ndarray, bins: int = 32) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256]).astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist[hist > 1e-12] / total
    return float(-(p * np.log2(p)).sum())


def compute_image_features_for_manifest(ctx: Dict[str, Any], manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for frame in manifest_df.to_dict(orient="records"):
        img = cv2.imread(frame["frame_path"])
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = cv2.Canny(gray, 50, 150)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        shifted = np.zeros_like(gray, dtype=np.float32)
        shifted[:, :-2] = gray[:, 2:].astype(np.float32)
        gray_f = gray.astype(np.float32)

        rows.append({
            "dataset": frame["dataset"],
            "material": frame["material"],
            "video_id": frame["video_id"],
            "frame_idx": frame["frame_idx"],
            "frame_key": frame["frame_key"],
            "lap_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "gray_entropy": _gray_entropy(gray),
            "color_entropy": _color_entropy(img),
            "tenengrad": float((gx ** 2 + gy ** 2).mean()),
            "brenner": float(((shifted - gray_f) ** 2).mean()),
            "edge_density": float(edges.astype(np.float32).mean() / 255.0),
            "local_contrast": float(np.mean(np.abs(gray_f - blur.astype(np.float32)))),
            "mean_luma": float(gray_f.mean()),
            "std_luma": float(gray_f.std()),
            "rms_contrast": float(gray_f.std() / (gray_f.mean() + 1e-6)),
        })

    out = pd.DataFrame(rows)
    write_parquet(out, artifact_path(ctx, "features", "image_features.parquet"))
    return out

