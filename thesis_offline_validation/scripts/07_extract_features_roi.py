#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.features_roi import build_roi_features
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ROI features from highest-confidence YOLOv8n detection")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        det_n_path = artifact_path(ctx, "inference", "detections_n.parquet")
        require_artifacts("07_extract_features_roi", [manifest_path, det_n_path])
        manifest_df = read_table(manifest_path)
        det_n = read_table(det_n_path)
        out = build_roi_features(ctx, manifest_df, det_n)
        record_stage_run(ctx, "07_extract_features_roi", {"rows": int(len(out)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
