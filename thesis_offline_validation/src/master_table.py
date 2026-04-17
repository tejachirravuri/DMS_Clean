from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .config import artifact_path
from .feature_registry import registry_df
from .io_utils import write_csv, write_parquet
from .schema_validation import assert_unique_rows, validate_feature_sets, validate_master_registry


KEYS = ["dataset", "material", "video_id", "frame_idx", "frame_key"]


def build_master_table(
    ctx: Dict[str, Any],
    manifest_df: pd.DataFrame,
    image_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    summary_n: pd.DataFrame,
    summary_s: pd.DataFrame,
    targets_df: pd.DataFrame,
) -> pd.DataFrame:
    targets_for_merge = targets_df.drop(columns=["n_infer_ms", "s_infer_ms"], errors="ignore").copy()
    n_out = summary_n.rename(columns={
        "infer_ms": "n_infer_ms",
        "det_count": "n_det_count_outcome",
        "mean_conf": "n_mean_conf_outcome",
        "max_conf": "n_max_conf_outcome",
        "conf_std": "n_conf_std_outcome",
        "total_area_norm": "n_total_area_norm_outcome",
        "zero_det_flag": "n_zero_det_flag_outcome",
    }).drop(columns=["model_tag"], errors="ignore")
    s_out = summary_s.rename(columns={
        "infer_ms": "s_infer_ms",
        "det_count": "s_det_count_outcome",
        "mean_conf": "s_mean_conf_outcome",
        "max_conf": "s_max_conf_outcome",
        "conf_std": "s_conf_std_outcome",
        "total_area_norm": "s_total_area_norm_outcome",
        "zero_det_flag": "s_zero_det_flag_outcome",
    }).drop(columns=["model_tag"], errors="ignore")

    validate_feature_sets(ctx.get("feature_config", {}))

    summary_keys = KEYS + ["timestamp_sec", "frame_path"]
    assert_unique_rows(manifest_df, KEYS, "master_table manifest")
    assert_unique_rows(image_df, KEYS, "master_table image features")
    assert_unique_rows(detection_df, KEYS, "master_table detection features")
    assert_unique_rows(temporal_df, KEYS, "master_table temporal features")
    assert_unique_rows(roi_df, KEYS, "master_table ROI features")
    assert_unique_rows(n_out, summary_keys, "master_table n summary")
    assert_unique_rows(s_out, summary_keys, "master_table s summary")
    assert_unique_rows(targets_for_merge, KEYS, "master_table targets")

    master = manifest_df.merge(image_df, on=KEYS, how="left", validate="one_to_one")
    master = master.merge(detection_df, on=KEYS, how="left", validate="one_to_one")
    master = master.merge(temporal_df, on=KEYS, how="left", validate="one_to_one")
    master = master.merge(roi_df, on=KEYS, how="left", validate="one_to_one")
    master = master.merge(n_out, on=summary_keys, how="left", validate="one_to_one")
    master = master.merge(s_out, on=summary_keys, how="left", validate="one_to_one")
    master = master.merge(targets_for_merge, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_target"))

    master["feature_version"] = ctx["config"].get("project", {}).get("feature_version", "v1")
    master["target_version"] = ctx["config"].get("project", {}).get("target_version", "v1")
    master["reference_mode"] = ctx["config"].get("project", {}).get("reference_mode", "s_pseudo_oracle")

    assert_unique_rows(master, KEYS, "master_table final")
    validate_master_registry(master)

    write_parquet(master, artifact_path(ctx, "master", "master_table.parquet"))
    write_csv(master.head(500), artifact_path(ctx, "master", "master_table_head.csv"))
    write_csv(registry_df(), artifact_path(ctx, "master", "feature_registry.csv"))
    return master
