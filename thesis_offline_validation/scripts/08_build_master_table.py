#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs
from src.master_table import build_master_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical master per-frame offline validation table")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-config", required=True)
    parser.add_argument("--targets-config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config, feature_config_path=args.feature_config, targets_config_path=args.targets_config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)

        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        image_path = artifact_path(ctx, "features", "image_features.parquet")
        detection_path = artifact_path(ctx, "features", "detection_features.parquet")
        temporal_path = artifact_path(ctx, "features", "temporal_features.parquet")
        summary_n_path = artifact_path(ctx, "inference", "frame_summary_n.parquet")
        summary_s_path = artifact_path(ctx, "inference", "frame_summary_s.parquet")
        targets_path = artifact_path(ctx, "targets", "targets.parquet")
        roi_path = artifact_path(ctx, "features", "roi_features.parquet")

        required_paths = [manifest_path, image_path, detection_path, temporal_path, summary_n_path, summary_s_path, targets_path]
        roi_enabled = ctx.get("feature_config", {}).get("feature_groups", {}).get("roi", {}).get("enabled", True)
        if roi_enabled:
            required_paths.append(roi_path)
        require_artifacts("08_build_master_table", required_paths)

        manifest_df = read_table(manifest_path)
        image_df = read_table(image_path)
        detection_df = read_table(detection_path)
        temporal_df = read_table(temporal_path)
        roi_df = read_table(roi_path) if roi_path.exists() else manifest_df[["dataset", "material", "video_id", "frame_idx", "frame_key"]].copy()
        summary_n = read_table(summary_n_path)
        summary_s = read_table(summary_s_path)
        targets_df = read_table(targets_path)

        master = build_master_table(ctx, manifest_df, image_df, detection_df, temporal_df, roi_df, summary_n, summary_s, targets_df)
        record_stage_run(ctx, "08_build_master_table", {"rows": int(len(master)), "columns": int(len(master.columns)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
