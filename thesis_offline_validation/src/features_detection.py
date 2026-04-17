from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .config import artifact_path
from .io_utils import write_parquet


def build_detection_features(ctx: Dict[str, Any], summary_n: pd.DataFrame) -> pd.DataFrame:
    cols = ["dataset", "material", "video_id", "frame_idx", "frame_key"]
    out = summary_n[cols + ["det_count", "mean_conf", "max_conf", "conf_std", "total_area_norm", "zero_det_flag"]].rename(columns={
        "det_count": "n_det_count",
        "mean_conf": "n_mean_conf",
        "max_conf": "n_max_conf",
        "conf_std": "n_conf_std",
        "total_area_norm": "n_total_area_norm",
        "zero_det_flag": "n_zero_det_flag",
    }).copy()
    write_parquet(out, artifact_path(ctx, "features", "detection_features.parquet"))
    return out

