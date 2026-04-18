from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summary(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _positive_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float((series == 1).mean())


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Each --run must be in the form label=/path/to/run_dir")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, path


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(run_dir / "manifest" / "frame_manifest.parquet")
    master = pd.read_parquet(run_dir / "master" / "master_table.parquet")
    targets = pd.read_parquet(run_dir / "targets" / "targets.parquet")
    return manifest, master, targets


def _per_video_rates(master: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        master.groupby("video_id")[column]
        .mean()
        .reset_index(name=f"{column}_positive_rate")
        .sort_values("video_id")
        .reset_index(drop=True)
    )


def report_run(label: str, run_dir: Path) -> None:
    manifest, master, targets = _load_run(run_dir)

    print(f"\n=== {label} ===")
    print(f"run_dir: {run_dir}")
    print("[A] Basic counts")
    print(f"manifest rows: {len(manifest)}")
    print(f"master rows  : {len(master)}")

    print("\n[B] Overall positive rates")
    print(f"y_hard_binary    : {_positive_rate(master['y_hard_binary']):.4f}")
    print(f"y_switch_optimal : {_positive_rate(master['y_switch_optimal']):.4f}")

    print("\n[C] Per-video positive rates: y_hard_binary")
    print(_per_video_rates(master, "y_hard_binary").to_string(index=False))
    print("\n[D] Per-video positive rates: y_switch_optimal")
    print(_per_video_rates(master, "y_switch_optimal").to_string(index=False))

    targets_cols = ["frame_key"]
    for col in ["extra_valid_dets_s", "matched_valid_pair_count"]:
        if col in master.columns:
            master[col] = master[col]
        elif col in targets.columns:
            targets_cols.append(col)
        else:
            print(f"\n[Notice] Column '{col}' not found in master_table.parquet or targets.parquet")

    merged = master
    if len(targets_cols) > 1:
        merged = master.merge(
            targets[targets_cols].drop_duplicates(subset=["frame_key"]),
            on="frame_key",
            how="left",
            validate="one_to_one",
        )

    print("\n[E] extra_valid_dets_s")
    if "extra_valid_dets_s" in merged.columns:
        print(_summary(merged["extra_valid_dets_s"]))
        print(f"frames with extra_valid_dets_s >= 1: {int((merged['extra_valid_dets_s'] >= 1).sum())}")
        print(f"frames with extra_valid_dets_s >= 2: {int((merged['extra_valid_dets_s'] >= 2).sum())}")

    print("\n[F] matched_valid_pair_count")
    if "matched_valid_pair_count" in merged.columns:
        print(_summary(merged["matched_valid_pair_count"]))
        print(f"frames with matched_valid_pair_count == 0: {int((merged['matched_valid_pair_count'] == 0).sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple target_v2 ablation runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        help="Run descriptor in the form label=/path/to/run_dir . Pass multiple times.",
    )
    args = parser.parse_args()

    for label, run_dir in args.run:
        report_run(label, run_dir)


if __name__ == "__main__":
    main()
