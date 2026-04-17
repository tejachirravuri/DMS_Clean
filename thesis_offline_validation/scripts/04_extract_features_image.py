#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import artifact_path, build_context
from src.features_image import compute_image_features_for_manifest
from src.io_utils import read_table, record_stage_run, require_artifacts, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image-only routing features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        manifest_path = artifact_path(ctx, "manifest", "frame_manifest.parquet")
        require_artifacts("04_extract_features_image", [manifest_path])
        manifest_df = read_table(manifest_path)
        out = compute_image_features_for_manifest(ctx, manifest_df)
        record_stage_run(ctx, "04_extract_features_image", {"rows": int(len(out)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
