#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import build_context, resolve_path
from src.frame_manifest import build_frame_manifest, discover_video_sources
from src.io_utils import record_stage_run, snapshot_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical frame manifest")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        ctx = build_context(args.config)
        missing = []
        discovered = discover_video_sources(ctx["config"])
        if not discovered:
            missing.append("no videos matched any configured source in offline_validation config")
        for source in ctx["config"].get("sources", []):
            video_dir = resolve_path(source["video_dir"])
            pattern = source.get("video_glob", "*.mp4")
            if not video_dir.exists():
                missing.append(f"video_dir missing: {video_dir}")
            elif not any(video_dir.glob(pattern)):
                missing.append(f"no videos matched pattern '{pattern}' in {video_dir}")
            if not resolve_path(source["weights_n"]).exists():
                missing.append(f"weights_n missing: {resolve_path(source['weights_n'])}")
            if not resolve_path(source["weights_s"]).exists():
                missing.append(f"weights_s missing: {resolve_path(source['weights_s'])}")
        if missing:
            raise FileNotFoundError("01_build_frame_manifest: missing required inputs:\n" + "\n".join(f"  - {item}" for item in missing))

        if ctx["config"].get("runtime", {}).get("save_config_snapshot", True):
            snapshot_configs(ctx)
        df = build_frame_manifest(ctx)
        record_stage_run(ctx, "01_build_frame_manifest", {"num_frames": int(len(df)), "cli_args": vars(args)})
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
