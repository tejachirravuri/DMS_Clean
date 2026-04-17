from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def build_context(
    config_path: str | Path,
    feature_config_path: Optional[str | Path] = None,
    targets_config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    cfg_path = resolve_path(config_path)
    cfg = load_yaml(cfg_path)
    feature_cfg = load_yaml(resolve_path(feature_config_path)) if feature_config_path else {}
    targets_cfg = load_yaml(resolve_path(targets_config_path)) if targets_config_path else {}

    run_name = cfg.get("project", {}).get("run_name", "default_run")
    output_root = resolve_path(cfg.get("paths", {}).get("output_root", "thesis_offline_validation/outputs"))
    run_dir = output_root / run_name

    return {
        "repo_root": repo_root(),
        "config_path": cfg_path,
        "feature_config_path": resolve_path(feature_config_path) if feature_config_path else None,
        "targets_config_path": resolve_path(targets_config_path) if targets_config_path else None,
        "config": cfg,
        "feature_config": feature_cfg,
        "targets_config": targets_cfg,
        "run_dir": run_dir,
    }


def artifact_dir(ctx: Dict[str, Any], name: str) -> Path:
    path = ctx["run_dir"] / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(ctx: Dict[str, Any], group: str, filename: str) -> Path:
    return artifact_dir(ctx, group) / filename

