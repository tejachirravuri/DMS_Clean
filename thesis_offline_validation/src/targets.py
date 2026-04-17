from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet


KEYS = ["dataset", "material", "video_id", "frame_idx", "frame_key"]


def _get_weights(cfg: Dict[str, Any], group: str) -> Dict[str, float]:
    return {k: float(v) for k, v in cfg.get("weights", {}).get(group, {}).items()}


def build_targets(
    ctx: Dict[str, Any],
    matching_df: pd.DataFrame,
    summary_n: pd.DataFrame,
    summary_s: pd.DataFrame,
) -> pd.DataFrame:
    target_cfg = ctx["targets_config"]
    reference_mode = target_cfg.get("reference_mode", ctx["config"].get("project", {}).get("reference_mode", "s_pseudo_oracle"))

    n_cols = summary_n.rename(columns={
        "infer_ms": "n_infer_ms",
        "det_count": "n_det_count_ref",
        "mean_conf": "n_mean_conf_ref",
        "max_conf": "n_max_conf_ref",
        "conf_std": "n_conf_std_ref",
        "total_area_norm": "n_total_area_norm_ref",
        "zero_det_flag": "n_zero_det_flag_ref",
    }).drop(columns=["model_tag", "timestamp_sec", "frame_path"], errors="ignore")
    s_cols = summary_s.rename(columns={
        "infer_ms": "s_infer_ms",
        "det_count": "s_det_count_ref",
        "mean_conf": "s_mean_conf_ref",
        "max_conf": "s_max_conf_ref",
        "conf_std": "s_conf_std_ref",
        "total_area_norm": "s_total_area_norm_ref",
        "zero_det_flag": "s_zero_det_flag_ref",
    }).drop(columns=["model_tag", "timestamp_sec", "frame_path"], errors="ignore")

    df = matching_df.merge(n_cols, on=KEYS, how="left").merge(s_cols, on=KEYS, how="left")

    df["extra_dets_s_norm"] = df["extra_dets_s"] / df["s_det_count_ref"].clip(lower=1)
    df["delta_infer_ms"] = df["s_infer_ms"] - df["n_infer_ms"]
    df["y_benefit_count"] = df["extra_dets_s"] - df["extra_dets_n"]
    df["y_benefit_conf"] = df["s_mean_conf_ref"] - df["n_mean_conf_ref"]
    df["y_benefit_iou"] = df["iou_disagreement"]

    disagree_w = _get_weights(target_cfg, "disagreement_score")
    df["y_disagreement_continuous"] = (
        disagree_w.get("det_count_gap", 0.30) * df["det_count_gap"].clip(lower=0)
        + disagree_w.get("mean_conf_gap", 0.25) * df["y_benefit_conf"].clip(lower=0.0)
        + disagree_w.get("iou_disagreement", 0.25) * df["iou_disagreement"]
        + disagree_w.get("extra_dets_s_norm", 0.20) * df["extra_dets_s_norm"]
    )

    dq_w = _get_weights(target_cfg, "delta_quality")
    df["delta_quality"] = (
        dq_w.get("extra_dets_s", 1.0) * df["extra_dets_s"]
        - dq_w.get("extra_dets_n", 0.5) * df["extra_dets_n"]
        + dq_w.get("mean_conf_gap", 0.5) * df["y_benefit_conf"].clip(lower=0.0)
        + dq_w.get("iou_disagreement", 0.5) * df["iou_disagreement"]
    )

    benefit_w = _get_weights(target_cfg, "benefit_score")
    df["benefit_score"] = (
        benefit_w.get("benefit_count", 1.0) * df["y_benefit_count"].clip(lower=0)
        + benefit_w.get("benefit_conf", 0.5) * df["y_benefit_conf"].clip(lower=0.0)
        + benefit_w.get("benefit_iou", 0.5) * df["y_benefit_iou"]
        + benefit_w.get("disagreement", 1.0) * df["y_disagreement_continuous"]
    )

    latency_w = _get_weights(target_cfg, "latency_penalty")
    df["latency_penalty"] = latency_w.get("delta_infer_ms", 0.01) * df["delta_infer_ms"].clip(lower=0.0)

    hard_cfg = target_cfg.get("targets", {}).get("hard_binary", {})
    min_extra = int(hard_cfg.get("min_extra_dets_s", 1))
    min_benefit = float(hard_cfg.get("min_benefit_score", 0.2))
    df["y_hard_binary"] = (
        (df["extra_dets_s"] >= min_extra) | (df["benefit_score"] >= min_benefit)
    ).astype(int)

    switch_cfg = target_cfg.get("targets", {}).get("switch_optimal", {})
    min_margin = float(switch_cfg.get("min_margin", 0.0))
    df["y_switch_optimal"] = (
        df["benefit_score"] > (df["latency_penalty"] + min_margin)
    ).astype(int)

    df["target_version"] = ctx["config"].get("project", {}).get("target_version", "v1")
    df["reference_mode"] = reference_mode
    write_parquet(df, artifact_path(ctx, "targets", "targets.parquet"))
    return df

