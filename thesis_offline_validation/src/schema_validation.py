from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

from .feature_registry import registry_df


IDENTITY_COLUMNS = {
    "dataset",
    "material",
    "video_id",
    "frame_idx",
    "frame_key",
    "timestamp_sec",
    "frame_path",
}

METADATA_COLUMNS = {
    "source_video_path",
    "weights_n",
    "weights_s",
    "fps",
    "width",
    "height",
    "frame_sha1",
    "feature_version",
    "target_version",
    "reference_mode",
}


def assert_unique_rows(df: pd.DataFrame, keys: list[str], label: str) -> None:
    if df.empty:
        return
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return
    dup_preview = df.loc[dup_mask, keys].head(10).to_dict(orient="records")
    raise ValueError(
        f"{label}: duplicate rows detected for join key(s) {keys}. "
        f"Sample duplicates: {dup_preview}"
    )


def _registry_index() -> pd.DataFrame:
    reg = registry_df().copy()
    if reg["column"].duplicated().any():
        dup_cols = reg.loc[reg["column"].duplicated(keep=False), "column"].tolist()
        raise ValueError(f"feature_registry: duplicate registry entries found: {dup_cols}")
    return reg.set_index("column", drop=False)


def validate_feature_sets(feature_cfg: Dict[str, Any]) -> None:
    reg = _registry_index()
    issues: list[str] = []
    seen: list[str] = []

    for group_name, group_cfg in feature_cfg.get("feature_groups", {}).items():
        if not group_cfg.get("enabled", True):
            continue
        for column in group_cfg.get("columns", []):
            seen.append(column)
            if column not in reg.index:
                issues.append(f"feature_sets.yaml: '{column}' in group '{group_name}' is not registered")
                continue
            row = reg.loc[column]
            if not bool(row["controller_allowed"]):
                issues.append(f"feature_sets.yaml: '{column}' in group '{group_name}' is not controller-allowed")
            if row["availability"] == "offline_only":
                issues.append(f"feature_sets.yaml: '{column}' in group '{group_name}' is offline_only")

    dup_features = sorted({column for column in seen if seen.count(column) > 1})
    if dup_features:
        issues.append(f"feature_sets.yaml: duplicate feature entries found: {dup_features}")

    if issues:
        raise ValueError("\n".join(issues))


def validate_master_registry(master_df: pd.DataFrame) -> None:
    reg = _registry_index()
    issues: list[str] = []

    missing = sorted(
        column
        for column in master_df.columns
        if column not in reg.index and column not in IDENTITY_COLUMNS and column not in METADATA_COLUMNS
    )
    if missing:
        issues.append(f"master_table: unregistered non-identity/non-metadata columns found: {missing}")

    offline_allowed = reg[(reg["availability"] == "offline_only") & (reg["controller_allowed"] == True)]
    if not offline_allowed.empty:
        issues.append(
            "feature_registry: offline_only columns cannot be controller_allowed=True: "
            f"{offline_allowed['column'].tolist()}"
        )

    target_allowed = reg[reg["column"].str.startswith("y_") & (reg["controller_allowed"] == True)]
    if not target_allowed.empty:
        issues.append(
            "feature_registry: target columns cannot be controller_allowed=True: "
            f"{target_allowed['column'].tolist()}"
        )

    sensitive_prefixes = ("s_", "matched_")
    sensitive_columns = {"mean_iou", "iou_disagreement", "det_count_gap"}
    extra_sensitive = reg[
        (
            reg["column"].str.startswith(sensitive_prefixes)
            | reg["column"].isin(sensitive_columns)
            | reg["column"].str.startswith("extra_dets_")
        )
        & (reg["controller_allowed"] == True)
    ]
    if not extra_sensitive.empty:
        issues.append(
            "feature_registry: s-derived or matching-derived columns cannot be controller_allowed=True: "
            f"{extra_sensitive['column'].tolist()}"
        )

    if issues:
        raise ValueError("\n".join(issues))
