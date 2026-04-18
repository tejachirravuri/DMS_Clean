from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet
from .matching import greedy_match


KEYS = ["dataset", "material", "video_id", "frame_idx", "frame_key"]
V2_SAVE_COLUMNS = [
    "dataset",
    "material",
    "video_id",
    "frame_idx",
    "frame_key",
    "n_valid_det_count",
    "s_valid_det_count",
    "extra_valid_dets_s",
    "extra_valid_dets_n",
    "matched_valid_pair_count",
    "matched_valid_conf_gain_mean",
    "valid_mean_iou",
    "y_localization_disagreement",
    "y_benefit_count",
    "y_benefit_conf",
    "y_benefit_iou",
    "y_disagreement_continuous",
    "benefit_score_v2",
    "delta_infer_ms",
    "latency_penalty",
    "y_hard_binary",
    "y_switch_optimal",
    "target_version",
    "reference_mode",
]


def _get_weights(cfg: Dict[str, Any], group: str) -> Dict[str, float]:
    return {k: float(v) for k, v in cfg.get("weights", {}).get(group, {}).items()}


def _get_target_version(ctx: Dict[str, Any]) -> str:
    return str(ctx["config"].get("project", {}).get("target_version", "v1"))


def _get_target_cfg(ctx: Dict[str, Any], version: str) -> Dict[str, Any]:
    base_cfg = ctx["targets_config"]
    if version == "v2":
        return base_cfg.get("versions", {}).get("v2", {})
    return base_cfg


def _apply_optional_valid_filters(frame_df: pd.DataFrame, valid_cfg: Dict[str, Any]) -> pd.DataFrame:
    if frame_df.empty:
        return frame_df.copy()

    out = frame_df.copy()
    tau_conf = float(valid_cfg.get("tau_conf_valid", 0.25))
    tau_area = float(valid_cfg.get("tau_area_min", 0.0005))
    out = out[(out["conf"] >= tau_conf) & (out["area_norm"] >= tau_area)].copy()

    top_k = int(valid_cfg.get("top_k", 0) or 0)
    if top_k > 0 and not out.empty:
        out = out.sort_values("conf", ascending=False).head(top_k).copy()

    geom_cfg = valid_cfg.get("geometry", {})
    if bool(geom_cfg.get("enabled", False)) and not out.empty:
        widths = (out["x2"] - out["x1"]).clip(lower=0.0)
        heights = (out["y2"] - out["y1"]).clip(lower=0.0)
        aspect = widths / heights.replace(0.0, np.nan)
        ar_min = float(geom_cfg.get("aspect_ratio_min", 0.0))
        ar_max = float(geom_cfg.get("aspect_ratio_max", np.inf))
        out = out[(widths > 0.0) & (heights > 0.0) & aspect.between(ar_min, ar_max, inclusive="both")].copy()

    return out


def _compute_v2_valid_matching(
    ctx: Dict[str, Any],
    matching_df: pd.DataFrame,
    det_n: pd.DataFrame,
    det_s: pd.DataFrame,
) -> pd.DataFrame:
    valid_cfg = _get_target_cfg(ctx, "v2").get("valid_detections", {})
    threshold = float(ctx["config"].get("matching", {}).get("iou_match_threshold", 0.5))

    det_n_groups = det_n.groupby("frame_key") if not det_n.empty else {}
    det_s_groups = det_s.groupby("frame_key") if not det_s.empty else {}
    rows = []

    for frame in matching_df.to_dict(orient="records"):
        frame_key = frame["frame_key"]
        frame_n = det_n_groups.get_group(frame_key) if not det_n.empty and frame_key in det_n_groups.groups else pd.DataFrame(columns=det_n.columns)
        frame_s = det_s_groups.get_group(frame_key) if not det_s.empty and frame_key in det_s_groups.groups else pd.DataFrame(columns=det_s.columns)

        valid_n = _apply_optional_valid_filters(frame_n, valid_cfg)
        valid_s = _apply_optional_valid_filters(frame_s, valid_cfg)
        matched, unmatched_n, unmatched_s = greedy_match(valid_n, valid_s, threshold)

        valid_ious = [m[2] for m in matched]
        conf_gains = []
        for n_idx, s_idx, _ in matched:
            conf_n = float(valid_n.loc[n_idx, "conf"])
            conf_s = float(valid_s.loc[s_idx, "conf"])
            conf_gains.append(max(conf_s - conf_n, 0.0))

        matched_valid_pair_count = int(len(matched))
        valid_mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0
        matched_valid_conf_gain_mean = float(np.mean(conf_gains)) if conf_gains else 0.0

        rows.append({
            "frame_key": frame_key,
            "n_valid_det_count": int(len(valid_n)),
            "s_valid_det_count": int(len(valid_s)),
            "extra_valid_dets_s": int(len(unmatched_s)),
            "extra_valid_dets_n": int(len(unmatched_n)),
            "matched_valid_pair_count": matched_valid_pair_count,
            "matched_valid_conf_gain_mean": matched_valid_conf_gain_mean,
            "valid_mean_iou": valid_mean_iou,
            "y_localization_disagreement": float(1.0 - valid_mean_iou) if matched_valid_pair_count > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def _build_targets_v1(
    ctx: Dict[str, Any],
    matching_df: pd.DataFrame,
    summary_n: pd.DataFrame,
    summary_s: pd.DataFrame,
) -> pd.DataFrame:
    target_cfg = _get_target_cfg(ctx, "v1")
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
    return df


def _build_targets_v2(
    ctx: Dict[str, Any],
    matching_df: pd.DataFrame,
    summary_n: pd.DataFrame,
    summary_s: pd.DataFrame,
    det_n: pd.DataFrame,
    det_s: pd.DataFrame,
) -> pd.DataFrame:
    target_cfg = _get_target_cfg(ctx, "v2")
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

    valid_match_df = _compute_v2_valid_matching(ctx, matching_df, det_n, det_s)
    df = matching_df.merge(n_cols, on=KEYS, how="left").merge(s_cols, on=KEYS, how="left")
    df = df.merge(valid_match_df, on="frame_key", how="left")

    fill_zero_cols = [
        "n_valid_det_count",
        "s_valid_det_count",
        "extra_valid_dets_s",
        "extra_valid_dets_n",
        "matched_valid_pair_count",
        "matched_valid_conf_gain_mean",
        "valid_mean_iou",
        "y_localization_disagreement",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    df["delta_infer_ms"] = df["s_infer_ms"] - df["n_infer_ms"]
    df["y_benefit_count"] = df["extra_valid_dets_s"] - df["extra_valid_dets_n"]
    df["y_benefit_conf"] = df["matched_valid_conf_gain_mean"]
    df["y_benefit_iou"] = df["y_localization_disagreement"]

    disagree_w = _get_weights(target_cfg, "disagreement_score")
    df["y_disagreement_continuous"] = (
        disagree_w.get("benefit_count", 1.0) * df["y_benefit_count"].clip(lower=0.0)
        + disagree_w.get("benefit_conf", 0.5) * df["y_benefit_conf"]
        + disagree_w.get("localization_disagreement", 0.5) * df["y_localization_disagreement"]
    )

    benefit_w = _get_weights(target_cfg, "benefit_score_v2")
    df["benefit_score_v2"] = (
        benefit_w.get("benefit_count", 1.0) * df["y_benefit_count"].clip(lower=0.0)
        + benefit_w.get("benefit_conf", 0.5) * df["y_benefit_conf"]
    )

    latency_w = _get_weights(target_cfg, "latency_penalty")
    df["latency_penalty"] = latency_w.get("delta_infer_ms", 0.01) * df["delta_infer_ms"].clip(lower=0.0)

    hard_cfg = target_cfg.get("targets", {}).get("hard_binary", {})
    min_extra = int(hard_cfg.get("min_extra_valid_dets_s", 1))
    tau_hard = float(hard_cfg.get("tau_hard", 0.50))
    df["y_hard_binary"] = (
        (df["extra_valid_dets_s"] >= min_extra) | (df["y_disagreement_continuous"] >= tau_hard)
    ).astype(int)

    switch_cfg = target_cfg.get("targets", {}).get("switch_optimal", {})
    min_margin = float(switch_cfg.get("min_margin", 0.10))
    df["y_switch_optimal"] = (
        df["benefit_score_v2"] > (df["latency_penalty"] + min_margin)
    ).astype(int)

    df["target_version"] = "v2"
    df["reference_mode"] = reference_mode

    for col in V2_SAVE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[V2_SAVE_COLUMNS].copy()


def build_targets(
    ctx: Dict[str, Any],
    matching_df: pd.DataFrame,
    summary_n: pd.DataFrame,
    summary_s: pd.DataFrame,
    det_n: pd.DataFrame | None = None,
    det_s: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_version = _get_target_version(ctx)
    if target_version == "v2":
        if det_n is None or det_s is None:
            raise ValueError("build_targets: target_version=v2 requires det_n and det_s inputs for valid-detection target construction")
        df = _build_targets_v2(ctx, matching_df, summary_n, summary_s, det_n, det_s)
    else:
        df = _build_targets_v1(ctx, matching_df, summary_n, summary_s)
    write_parquet(df, artifact_path(ctx, "targets", "targets.parquet"))
    return df
