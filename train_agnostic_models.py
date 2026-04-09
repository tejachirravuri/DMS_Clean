#!/usr/bin/env python3
"""
Train YOLO11 and YOLOv8m models on insulator dataset for DMS agnosticism experiments.
Run on GPU for fast training.

Usage (on remote GPU):
    python train_agnostic_models.py --dataset glass --device 0
    python train_agnostic_models.py --dataset porcelain --device 0
    python train_agnostic_models.py --dataset glass --device 0 --models yolov8m
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Train agnostic models for DMS")
    parser.add_argument("--dataset", choices=["glass", "porcelain"], required=True)
    parser.add_argument("--device", default="0", help="GPU device (0, 1, cpu)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--models", nargs="+",
                        default=["yolo11n", "yolo11s", "yolov8m"],
                        help="Models to train")
    parser.add_argument("--data-yaml", default=None,
                        help="Path to data.yaml (auto-detected if not set)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Auto-detect data.yaml
    if args.data_yaml:
        data_yaml = args.data_yaml
    else:
        # Try common locations
        candidates = [
            f"datasets/{args.dataset}_insulator.yaml",
            f"datasets/{args.dataset}_MPID_dataset/{args.dataset}_insulator.yaml",
            f"{args.dataset}_insulator.yaml",
        ]
        data_yaml = None
        for c in candidates:
            if os.path.exists(c):
                data_yaml = c
                break
        if not data_yaml:
            print(f"ERROR: Cannot find data.yaml for {args.dataset}.")
            print(f"Tried: {candidates}")
            print(f"Use --data-yaml to specify manually.")
            sys.exit(1)

    print(f"=" * 60)
    print(f"DMS Agnosticism Model Training")
    print(f"=" * 60)
    print(f"Dataset:  {args.dataset}")
    print(f"Data:     {data_yaml}")
    print(f"Device:   {args.device}")
    print(f"Models:   {args.models}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch}")
    print(f"=" * 60)

    project_dir = f"agnostic_training/{args.dataset}"

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Training: {model_name} on {args.dataset}")
        print(f"{'='*60}")

        # Map model name to pretrained weights
        weight_map = {
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolov8m": "yolov8m.pt",
            "yolov8l": "yolov8l.pt",
        }
        pretrained = weight_map.get(model_name, f"{model_name}.pt")

        # Download pretrained model (COCO)
        print(f"Loading pretrained: {pretrained}")
        model = YOLO(pretrained)

        # Train
        run_name = f"{model_name}_{args.dataset}"
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            patience=20,  # early stopping
            save=True,
            plots=True,
            verbose=True,
        )

        # Report
        best_path = Path(project_dir) / run_name / "weights" / "best.pt"
        if best_path.exists():
            size_mb = best_path.stat().st_size / (1024 * 1024)
            print(f"\n[OK] Saved: {best_path} ({size_mb:.1f} MB)")
        else:
            print(f"\n[WARN] best.pt not found at {best_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nTrained model weights are at:")
    for model_name in args.models:
        run_name = f"{model_name}_{args.dataset}"
        p = Path(project_dir) / run_name / "weights" / "best.pt"
        exists = "EXISTS" if p.exists() else "MISSING"
        print(f"  {p}  [{exists}]")

    print(f"\nNext step: Copy best.pt files to DMS model pairs and run:")
    print(f"  python dms_experiment.py overnight --models-dir <pair_dir> --videos-dir videos --device cpu")


if __name__ == "__main__":
    main()
