#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.detector_runner import run_dual_inference
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8n and YOLOv8s on manifest frames")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        require_artifacts("02_run_dual_inference", [manifest_path])
        manifest_df = read_table(manifest_path)
        extra_required = [Path(path) for path in manifest_df["frame_path"].dropna().unique().tolist()]
        extra_required.extend(Path(path) for path in manifest_df["weights_n"].dropna().unique().tolist())
        extra_required.extend(Path(path) for path in manifest_df["weights_s"].dropna().unique().tolist())
        require_artifacts("02_run_dual_inference", extra_required)
        det_n, det_s, sum_n, sum_s = run_dual_inference(ctx, manifest_df)
        record_stage_run(
            ctx,
            "02_run_dual_inference",
            {
                "frames": int(len(manifest_df)),
                "detections_n": int(len(det_n)),
                "detections_s": int(len(det_s)),
                "summary_n": int(len(sum_n)),
                "summary_s": int(len(sum_s)),
                "cli_args": vars(args),
            },
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
