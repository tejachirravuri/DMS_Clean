#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.features_temporal import build_temporal_features
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract temporal routing features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        image_path = artifact_path(ctx, "features", "image_features.parquet")
        detection_path = artifact_path(ctx, "features", "detection_features.parquet")
        require_artifacts("06_extract_features_temporal", [manifest_path, image_path, detection_path])
        manifest_df = read_table(manifest_path)
        image_df = read_table(image_path)
        detection_df = read_table(detection_path)
        out = build_temporal_features(ctx, manifest_df, image_df, detection_df)
        record_stage_run(ctx, "06_extract_features_temporal", {"rows": int(len(out)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
