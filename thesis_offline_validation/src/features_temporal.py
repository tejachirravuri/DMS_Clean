from __future__ import annotations

import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet


def build_temporal_features(ctx, manifest_df: pd.DataFrame, image_df: pd.DataFrame, detection_df: pd.DataFrame) -> pd.DataFrame:
    base = manifest_df[["dataset", "material", "video_id", "frame_idx", "frame_key"]].merge(
        image_df[["frame_key", "lap_var", "gray_entropy"]], on="frame_key", how="left"
    ).merge(
        detection_df[["frame_key", "n_mean_conf", "n_det_count", "n_zero_det_flag"]], on="frame_key", how="left"
    )
    base = base.sort_values(["dataset", "material", "video_id", "frame_idx"]).reset_index(drop=True)

    groups = []
    for _, g in base.groupby(["dataset", "material", "video_id"], sort=False):
        g = g.copy()
        g["lap_var_delta_1"] = g["lap_var"].diff().fillna(0.0)
        g["gray_entropy_delta_1"] = g["gray_entropy"].diff().fillna(0.0)
        g["n_mean_conf_delta_1"] = g["n_mean_conf"].diff().fillna(0.0)
        g["n_det_count_delta_1"] = g["n_det_count"].diff().fillna(0.0)
        g["n_mean_conf_ema_fast"] = g["n_mean_conf"].ewm(alpha=0.30, adjust=False).mean()
        g["n_mean_conf_ema_slow"] = g["n_mean_conf"].ewm(alpha=0.02, adjust=False).mean()
        g["conf_drop"] = (g["n_mean_conf_ema_slow"] - g["n_mean_conf_ema_fast"]).clip(lower=0.0) / (
            g["n_mean_conf_ema_slow"].abs() + 1e-9
        )
        streak = []
        count = 0
        for flag in g["n_zero_det_flag"].fillna(0).astype(int):
            count = count + 1 if flag == 1 else 0
            streak.append(count)
        g["zero_det_streak"] = streak
        groups.append(g)

    out = pd.concat(groups, ignore_index=True) if groups else base.iloc[0:0].copy()
    cols = [
        "dataset", "material", "video_id", "frame_idx", "frame_key",
        "lap_var_delta_1", "gray_entropy_delta_1", "n_mean_conf_delta_1",
        "n_det_count_delta_1", "n_mean_conf_ema_fast", "n_mean_conf_ema_slow",
        "conf_drop", "zero_det_streak",
    ]
    out = out[cols]
    write_parquet(out, artifact_path(ctx, "features", "temporal_features.parquet"))
    return out

