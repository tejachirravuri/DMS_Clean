#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.features_detection import build_detection_features
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLOv8n detection-state features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        summary_n_path = artifact_path(ctx, "inference", "frame_summary_n.parquet")
        require_artifacts("05_extract_features_detection", [summary_n_path])
        summary_n = read_table(summary_n_path)
        out = build_detection_features(ctx, summary_n)
        record_stage_run(ctx, "05_extract_features_detection", {"rows": int(len(out)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
