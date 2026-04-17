from pathlib import Path
import pandas as pd

run_dir = Path("thesis_offline_validation/outputs/smoke_test_v1")

print("\n[1] Artifact existence")
expected = [
    run_dir / "manifest/frame_manifest.parquet",
    run_dir / "manifest/frame_manifest.csv",
    run_dir / "inference/detections_n.parquet",
    run_dir / "inference/detections_s.parquet",
    run_dir / "inference/frame_summary_n.parquet",
    run_dir / "inference/frame_summary_s.parquet",
    run_dir / "targets/frame_matching.parquet",
    run_dir / "targets/targets.parquet",
    run_dir / "features/image_features.parquet",
    run_dir / "features/detection_features.parquet",
    run_dir / "features/temporal_features.parquet",
    run_dir / "features/roi_features.parquet",
    run_dir / "master/master_table.parquet",
    run_dir / "master/master_table_head.csv",
    run_dir / "master/feature_registry.csv",
]
missing = [str(p) for p in expected if not p.exists()]
if missing:
    print("Missing artifacts:")
    for p in missing:
        print("  ", p)
else:
    print("All expected artifacts exist.")

manifest = pd.read_parquet(run_dir / "manifest/frame_manifest.parquet")
sum_n = pd.read_parquet(run_dir / "inference/frame_summary_n.parquet")
sum_s = pd.read_parquet(run_dir / "inference/frame_summary_s.parquet")
matching = pd.read_parquet(run_dir / "targets/frame_matching.parquet")
targets = pd.read_parquet(run_dir / "targets/targets.parquet")
img = pd.read_parquet(run_dir / "features/image_features.parquet")
det = pd.read_parquet(run_dir / "features/detection_features.parquet")
tmp = pd.read_parquet(run_dir / "features/temporal_features.parquet")
roi = pd.read_parquet(run_dir / "features/roi_features.parquet")
master = pd.read_parquet(run_dir / "master/master_table.parquet")
registry = pd.read_csv(run_dir / "master/feature_registry.csv")

print("\n[2] Row counts")
print("manifest :", len(manifest))
print("summary_n:", len(sum_n))
print("summary_s:", len(sum_s))
print("matching :", len(matching))
print("targets  :", len(targets))
print("image    :", len(img))
print("detect   :", len(det))
print("temporal :", len(tmp))
print("roi      :", len(roi))
print("master   :", len(master))

print("\n[3] Consistency checks")
ok = True
base_n = len(manifest)
for name, df in [
    ("summary_n", sum_n),
    ("summary_s", sum_s),
    ("matching", matching),
    ("targets", targets),
    ("image", img),
    ("detect", det),
    ("temporal", tmp),
    ("roi", roi),
    ("master", master),
]:
    if len(df) != base_n:
        print(f"Row mismatch: {name}={len(df)} vs manifest={base_n}")
        ok = False
if ok:
    print("Per-frame tables are row-consistent with manifest.")

dup = master.duplicated(subset=["dataset", "material", "video_id", "frame_idx", "frame_key"]).sum()
print("\n[4] One-row-per-frame invariant")
print("duplicate master rows:", int(dup))
if dup == 0 and len(master) == len(manifest):
    print("Master table is one row per frame.")
else:
    print("Master table violates one-row-per-frame invariant.")

print("\n[5] NaN check in key columns")
key_cols = [
    "lap_var", "gray_entropy", "n_det_count", "n_mean_conf",
    "lap_var_delta_1", "gray_entropy_delta_1",
    "y_hard_binary", "y_benefit_count", "y_benefit_conf",
    "y_benefit_iou", "y_disagreement_continuous", "y_switch_optimal",
]
for col in key_cols:
    if col in master.columns:
        n_nan = int(master[col].isna().sum())
        print(f"{col}: NaNs={n_nan}/{len(master)}")

print("\n[6] Target population")
target_cols = [
    "y_hard_binary", "y_benefit_count", "y_benefit_conf",
    "y_benefit_iou", "y_disagreement_continuous", "y_switch_optimal",
]
for col in target_cols:
    if col in master.columns:
        nunique = master[col].nunique(dropna=True)
        print(f"{col}: non-null={int(master[col].notna().sum())}, unique={int(nunique)}")

print("\n[7] Feature registry export")
print("registry rows:", len(registry))
print("registry exists:", (run_dir / "master/feature_registry.csv").exists())

print("\n[8] Smoke verdict")
checks = [
    not missing,
    len(master) == len(manifest),
    dup == 0,
    all(col in master.columns for col in target_cols),
    (run_dir / "master/feature_registry.csv").exists(),
]
print("PASS" if all(checks) else "WARNING")