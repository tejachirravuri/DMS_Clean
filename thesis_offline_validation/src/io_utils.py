from __future__ import annotations

import json
import os
import platform
import shutil
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .config import artifact_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_csv(path, index=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def require_artifacts(stage_name: str, paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"{stage_name}: missing required input artifact(s):\n{missing_text}")


def snapshot_configs(ctx: Dict[str, Any]) -> None:
    snap_dir = artifact_dir(ctx, "snapshots")
    shutil.copy2(ctx["config_path"], snap_dir / Path(ctx["config_path"]).name)
    if ctx.get("feature_config_path"):
        shutil.copy2(ctx["feature_config_path"], snap_dir / Path(ctx["feature_config_path"]).name)
    if ctx.get("targets_config_path"):
        shutil.copy2(ctx["targets_config_path"], snap_dir / Path(ctx["targets_config_path"]).name)


def collect_environment_info() -> Dict[str, Any]:
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "opencv-python", "PyYAML", "ultralytics"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp_utc": utc_now_iso(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "packages": packages,
    }


def record_stage_run(ctx: Dict[str, Any], stage_name: str, extra: Dict[str, Any] | None = None) -> None:
    payload = {
        "stage": stage_name,
        "run_name": ctx["config"].get("project", {}).get("run_name", "default_run"),
        "config_path": str(ctx["config_path"]),
        "feature_config_path": str(ctx["feature_config_path"]) if ctx.get("feature_config_path") else None,
        "targets_config_path": str(ctx["targets_config_path"]) if ctx.get("targets_config_path") else None,
        "environment": collect_environment_info(),
    }
    if extra:
        payload.update(extra)
    write_json(artifact_dir(ctx, "runs") / f"{stage_name}_run_manifest.json", payload)
