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
from src.matching import build_frame_matching
from src.targets import build_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frame-level matching summaries and targets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets-config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config, targets_config_path=args.targets_config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)

        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        det_n_path = artifact_path(ctx, "inference", "detections_n.parquet")
        det_s_path = artifact_path(ctx, "inference", "detections_s.parquet")
        sum_n_path = artifact_path(ctx, "inference", "frame_summary_n.parquet")
        sum_s_path = artifact_path(ctx, "inference", "frame_summary_s.parquet")
        require_artifacts("03_build_targets", [manifest_path, det_n_path, det_s_path, sum_n_path, sum_s_path])

        manifest_df = read_table(manifest_path)
        det_n = read_table(det_n_path)
        det_s = read_table(det_s_path)
        sum_n = read_table(sum_n_path)
        sum_s = read_table(sum_s_path)

        matching_df = build_frame_matching(ctx, manifest_df, det_n, det_s)
        targets_df = build_targets(ctx, matching_df, sum_n, sum_s)
        record_stage_run(
            ctx,
            "03_build_targets",
            {"matching_rows": int(len(matching_df)), "target_rows": int(len(targets_df)), "cli_args": vars(args)},
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
