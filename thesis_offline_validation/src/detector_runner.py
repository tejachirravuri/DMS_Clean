from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from ultralytics import YOLO

from .config import artifact_path
from .io_utils import write_parquet


def _summarize_frame_detections(frame_row: Dict[str, Any], model_tag: str, dets: List[Dict[str, Any]], infer_ms: float) -> Dict[str, Any]:
    confs = [d["conf"] for d in dets]
    total_area = sum(d["area_norm"] for d in dets)
    return {
        "dataset": frame_row["dataset"],
        "material": frame_row["material"],
        "video_id": frame_row["video_id"],
        "frame_idx": frame_row["frame_idx"],
        "frame_key": frame_row["frame_key"],
        "timestamp_sec": frame_row["timestamp_sec"],
        "frame_path": frame_row["frame_path"],
        "model_tag": model_tag,
        "infer_ms": float(infer_ms),
        "det_count": int(len(dets)),
        "mean_conf": float(sum(confs) / len(confs)) if confs else 0.0,
        "max_conf": float(max(confs)) if confs else 0.0,
        "conf_std": float(pd.Series(confs).std(ddof=0)) if len(confs) > 1 else 0.0,
        "total_area_norm": float(total_area),
        "zero_det_flag": int(len(dets) == 0),
    }


def run_inference_for_model(manifest_df: pd.DataFrame, model_path: str, model_tag: str, infer_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = YOLO(model_path)
    det_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for frame in manifest_df.to_dict(orient="records"):
        t0 = time.perf_counter()
        res = model.predict(
            source=frame["frame_path"],
            imgsz=int(infer_cfg.get("imgsz", 640)),
            device=infer_cfg.get("device", "cpu"),
            conf=float(infer_cfg.get("conf", 0.001)),
            iou=float(infer_cfg.get("iou", 0.45)),
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dets_frame: List[Dict[str, Any]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            frame_area = max(1.0, float(frame["width"] * frame["height"]))
            for det_id, (box, c_val, cls_id) in enumerate(zip(xyxy, conf, cls)):
                x1, y1, x2, y2 = map(float, box[:4])
                area_norm = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
                det_row = {
                    "dataset": frame["dataset"],
                    "material": frame["material"],
                    "video_id": frame["video_id"],
                    "frame_idx": frame["frame_idx"],
                    "frame_key": frame["frame_key"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "frame_path": frame["frame_path"],
                    "model_tag": model_tag,
                    "det_id": det_id,
                    "cls_id": int(cls_id),
                    "conf": float(c_val),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "area_norm": float(area_norm),
                }
                det_rows.append(det_row)
                dets_frame.append(det_row)
        summary_rows.append(_summarize_frame_detections(frame, model_tag, dets_frame, infer_ms))

    return pd.DataFrame(det_rows), pd.DataFrame(summary_rows)


def run_dual_inference(ctx: Dict[str, Any], manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    infer_cfg = ctx["config"].get("inference", {})
    if manifest_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    det_n_parts = []
    det_s_parts = []
    sum_n_parts = []
    sum_s_parts = []
    group_cols = ["weights_n", "weights_s", "material"]
    for _, group_df in manifest_df.groupby(group_cols, sort=False):
        model_n_path = str(group_df["weights_n"].iloc[0])
        model_s_path = str(group_df["weights_s"].iloc[0])
        group_det_n, group_sum_n = run_inference_for_model(group_df, model_n_path, "n", infer_cfg)
        group_det_s, group_sum_s = run_inference_for_model(group_df, model_s_path, "s", infer_cfg)
        det_n_parts.append(group_det_n)
        det_s_parts.append(group_det_s)
        sum_n_parts.append(group_sum_n)
        sum_s_parts.append(group_sum_s)

    det_n = pd.concat(det_n_parts, ignore_index=True) if det_n_parts else pd.DataFrame()
    det_s = pd.concat(det_s_parts, ignore_index=True) if det_s_parts else pd.DataFrame()
    sum_n = pd.concat(sum_n_parts, ignore_index=True) if sum_n_parts else pd.DataFrame()
    sum_s = pd.concat(sum_s_parts, ignore_index=True) if sum_s_parts else pd.DataFrame()

    write_parquet(det_n, artifact_path(ctx, "inference", "detections_n.parquet"))
    write_parquet(det_s, artifact_path(ctx, "inference", "detections_s.parquet"))
    write_parquet(sum_n, artifact_path(ctx, "inference", "frame_summary_n.parquet"))
    write_parquet(sum_s, artifact_path(ctx, "inference", "frame_summary_s.parquet"))
    return det_n, det_s, sum_n, sum_s
