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


def build_roi_features(ctx, manifest_df: pd.DataFrame, det_n: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    best_n = det_n.sort_values(["frame_key", "conf"], ascending=[True, False]).drop_duplicates("frame_key")
    best_map = {row["frame_key"]: row for row in best_n.to_dict(orient="records")}

    for frame in manifest_df.to_dict(orient="records"):
        item = {
            "dataset": frame["dataset"],
            "material": frame["material"],
            "video_id": frame["video_id"],
            "frame_idx": frame["frame_idx"],
            "frame_key": frame["frame_key"],
            "roi_best_conf": np.nan,
            "roi_best_area_norm": np.nan,
            "roi_lap_var": np.nan,
            "roi_gray_entropy": np.nan,
        }
        det = best_map.get(frame["frame_key"])
        if det is not None:
            img = cv2.imread(frame["frame_path"])
            if img is not None:
                h, w = img.shape[:2]
                x1 = max(0, min(w - 1, int(det["x1"])))
                y1 = max(0, min(h - 1, int(det["y1"])))
                x2 = max(x1 + 1, min(w, int(det["x2"])))
                y2 = max(y1 + 1, min(h, int(det["y2"])))
                roi = img[y1:y2, x1:x2]
                if roi.size > 0 and roi.shape[0] >= 5 and roi.shape[1] >= 5:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    item.update({
                        "roi_best_conf": float(det["conf"]),
                        "roi_best_area_norm": float(det["area_norm"]),
                        "roi_lap_var": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                        "roi_gray_entropy": _gray_entropy(gray),
                    })
        rows.append(item)

    out = pd.DataFrame(rows)
    write_parquet(out, artifact_path(ctx, "features", "roi_features.parquet"))
    return out

