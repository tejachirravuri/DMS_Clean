from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd

from .config import artifact_path, resolve_path
from .io_utils import write_csv, write_parquet


def slugify(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return clean or "video"


def discover_video_sources(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = []
    for source in cfg.get("sources", []):
        video_dir = resolve_path(source["video_dir"])
        pattern = source.get("video_glob", "*.mp4")
        for video_path in sorted(video_dir.glob(pattern)):
            item = dict(source)
            item["video_path"] = video_path
            item["video_id"] = slugify(video_path.stem)
            sources.append(item)
    return sources


def build_frame_manifest(ctx: Dict[str, Any]) -> pd.DataFrame:
    cfg = ctx["config"]
    frame_cfg = cfg.get("frame_sampling", {})
    stride = int(frame_cfg.get("stride", 1))
    max_frames = int(frame_cfg.get("max_frames_per_video", 0))
    jpeg_quality = int(frame_cfg.get("jpeg_quality", 95))
    overwrite = bool(frame_cfg.get("overwrite_existing_frames", False))

    rows: List[Dict[str, Any]] = []
    frame_root = ctx["run_dir"] / cfg.get("paths", {}).get("frame_dirname", "frames")
    frame_root.mkdir(parents=True, exist_ok=True)

    for source in discover_video_sources(cfg):
        video_path = source["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        dataset = source["dataset"]
        material = source["material"]
        video_id = source["video_id"]
        out_dir = frame_root / dataset / material / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        raw_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and (raw_idx % stride) != 0:
                raw_idx += 1
                continue
            frame_name = f"frame_{raw_idx:06d}.jpg"
            frame_path = out_dir / frame_name
            if overwrite or not frame_path.exists():
                cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            timestamp_sec = raw_idx / fps if fps > 0 else 0.0
            frame_key = f"{dataset}__{material}__{video_id}__{raw_idx:06d}"
            rows.append({
                "dataset": dataset,
                "material": material,
                "video_id": video_id,
                "frame_idx": raw_idx,
                "frame_key": frame_key,
                "timestamp_sec": timestamp_sec,
                "frame_path": str(frame_path),
                "source_video_path": str(video_path),
                "weights_n": str(resolve_path(source["weights_n"])),
                "weights_s": str(resolve_path(source["weights_s"])),
                "fps": fps,
                "width": width,
                "height": height,
                "frame_sha1": hashlib.sha1(frame_key.encode("utf-8")).hexdigest(),
            })
            kept += 1
            raw_idx += 1
            if max_frames > 0 and kept >= max_frames:
                break
        cap.release()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["dataset", "material", "video_id", "frame_idx"]).reset_index(drop=True)
    write_parquet(df, artifact_path(ctx, "manifest", "frame_manifest.parquet"))
    write_csv(df, artifact_path(ctx, "manifest", "frame_manifest.csv"))
    return df

