from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(frame_n: pd.DataFrame, frame_s: pd.DataFrame, iou_threshold: float) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if frame_n.empty or frame_s.empty:
        return [], list(frame_n.index), list(frame_s.index)

    iou_matrix = np.zeros((len(frame_n), len(frame_s)), dtype=np.float32)
    n_boxes = frame_n[["x1", "y1", "x2", "y2"]].to_numpy()
    s_boxes = frame_s[["x1", "y1", "x2", "y2"]].to_numpy()
    for i in range(len(frame_n)):
        for j in range(len(frame_s)):
            iou_matrix[i, j] = box_iou_xyxy(n_boxes[i], s_boxes[j])

    matched: List[Tuple[int, int, float]] = []
    used_n, used_s = set(), set()
    while iou_matrix.size > 0:
        max_iou = float(iou_matrix.max())
        if max_iou < iou_threshold:
            break
        idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        i, j = int(idx[0]), int(idx[1])
        matched.append((frame_n.index[i], frame_s.index[j], max_iou))
        used_n.add(frame_n.index[i])
        used_s.add(frame_s.index[j])
        iou_matrix[i, :] = 0.0
        iou_matrix[:, j] = 0.0
    unmatched_n = [idx for idx in frame_n.index if idx not in used_n]
    unmatched_s = [idx for idx in frame_s.index if idx not in used_s]
    return matched, unmatched_n, unmatched_s


def build_frame_matching(ctx: Dict[str, Any], manifest_df: pd.DataFrame, det_n: pd.DataFrame, det_s: pd.DataFrame) -> pd.DataFrame:
    threshold = float(ctx["config"].get("matching", {}).get("iou_match_threshold", 0.5))
    rows: List[Dict[str, float | int | str]] = []

    det_n_groups = det_n.groupby("frame_key") if not det_n.empty else {}
    det_s_groups = det_s.groupby("frame_key") if not det_s.empty else {}

    for frame in manifest_df.to_dict(orient="records"):
        frame_key = frame["frame_key"]
        frame_n = det_n_groups.get_group(frame_key) if not det_n.empty and frame_key in det_n_groups.groups else pd.DataFrame(columns=det_n.columns)
        frame_s = det_s_groups.get_group(frame_key) if not det_s.empty and frame_key in det_s_groups.groups else pd.DataFrame(columns=det_s.columns)
        matched, unmatched_n, unmatched_s = greedy_match(frame_n, frame_s, threshold)
        mean_iou = float(np.mean([m[2] for m in matched])) if matched else 0.0
        s_det_count = int(len(frame_s))
        n_det_count = int(len(frame_n))
        extra_s = int(len(unmatched_s))
        extra_n = int(len(unmatched_n))
        matched_fraction = len(matched) / max(1, s_det_count)
        iou_disagreement = 1.0 - mean_iou if (n_det_count > 0 or s_det_count > 0) else 0.0

        rows.append({
            "dataset": frame["dataset"],
            "material": frame["material"],
            "video_id": frame["video_id"],
            "frame_idx": frame["frame_idx"],
            "frame_key": frame_key,
            "matched_count": int(len(matched)),
            "matched_fraction_s": float(matched_fraction),
            "mean_iou": float(mean_iou),
            "iou_disagreement": float(iou_disagreement),
            "extra_dets_s": extra_s,
            "extra_dets_n": extra_n,
            "det_count_gap": int(s_det_count - n_det_count),
        })

    out = pd.DataFrame(rows)
    write_parquet(out, artifact_path(ctx, "targets", "frame_matching.parquet"))
    return out

