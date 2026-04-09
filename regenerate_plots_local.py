#!/usr/bin/env python3
"""
Regenerate all plots locally from existing 12-video results.
Uses the fixed timing trace code (y-axis cap, reference lines, no 50-frame avg).

Usage:
    python regenerate_plots_local.py
"""

import json
import csv
import os
from pathlib import Path

# Check matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib/numpy not installed")
    exit(1)

RESULTS_DIR = Path("G:/DMS_Experiment_Results")
FIGURES_DIR = RESULTS_DIR / "figures_v2"
FIGURES_DIR.mkdir(exist_ok=True)

POLICIES = ["n_only", "s_only", "entropy_only", "combined", "combined_hyst",
            "conf_ema", "niqe_switch", "multi_proxy"]
SWITCHING = ["entropy_only", "combined", "combined_hyst", "conf_ema", "niqe_switch", "multi_proxy"]

POLICY_COLORS = {
    "n_only": "#2196F3", "s_only": "#F44336", "entropy_only": "#4CAF50",
    "combined": "#FF9800", "combined_hyst": "#9C27B0", "conf_ema": "#00BCD4",
    "niqe_switch": "#795548", "multi_proxy": "#607D8B",
}

def load_trace(csv_path):
    """Load a timing trace CSV."""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row)
    return frames


def plot_timing_timeseries(trace_csv, out_path, title=""):
    """Fixed timing trace plot with y-axis cap and reference lines."""
    frames = load_trace(trace_csv)
    if not frames:
        return

    frame_idx = list(range(len(frames)))
    t_total = [float(r.get("T_total_ms", r.get("T_total", 0))) for r in frames]
    choices = [r.get("choice", r.get("model", "n")) for r in frames]

    t_arr = np.array(t_total)
    mean_val = np.mean(t_arr)
    median_val = np.median(t_arr)
    p95 = np.percentile(t_arr, 95)
    p99 = np.percentile(t_arr, 99)
    outliers = np.sum(t_arr > p99 * 1.3)

    # Y-axis cap
    y_max = max(p99 * 1.3, p95 * 1.5, 50.0)

    # Colors by choice
    colors = ['#F44336' if c == 's' else '#2196F3' for c in choices]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(frame_idx, t_total, c=colors, s=3, alpha=0.4, edgecolors='none')

    # Reference lines
    ax.axhline(y=mean_val, color='green', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Mean: {mean_val:.1f}ms')
    ax.axhline(y=median_val, color='blue', linestyle=':', linewidth=1, alpha=0.7,
               label=f'Median: {median_val:.1f}ms')
    ax.axhline(y=p95, color='orange', linestyle=':', linewidth=1, alpha=0.7,
               label=f'P95: {p95:.1f}ms')

    ax.set_ylim(0, y_max)
    ax.set_xlabel('Frame')
    ax.set_ylabel('T_total (ms)')
    ax.set_title(f'{title}\nMean={mean_val:.1f}ms, Med={median_val:.1f}ms, P95={p95:.1f}ms, Outliers(>{p99*1.3:.0f}ms)={outliers}',
                 fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_comparison_bars(video_name, comparison_data, out_path):
    """Bar chart comparing all policies for a video."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    policies = [p for p in POLICIES if p in comparison_data]
    t_means = [comparison_data[p]["T_total_mean"] for p in policies]
    slow_pcts = [comparison_data[p]["slow_pct"] for p in policies]
    sw_per_100 = [comparison_data[p]["sw_per_100"] for p in policies]
    colors = [POLICY_COLORS.get(p, '#999') for p in policies]

    # T_mean bars
    axes[0].barh(policies, t_means, color=colors)
    axes[0].set_xlabel('T_total mean (ms)')
    axes[0].set_title('Latency')
    for i, v in enumerate(t_means):
        axes[0].text(v + 1, i, f'{v:.1f}', va='center', fontsize=8)

    # s-usage bars
    axes[1].barh(policies, slow_pcts, color=colors)
    axes[1].set_xlabel('s-route %')
    axes[1].set_title('s-Model Usage')
    axes[1].set_xlim(0, 105)

    # Switches bars
    axes[2].barh(policies, sw_per_100, color=colors)
    axes[2].set_xlabel('Switches per 100 frames')
    axes[2].set_title('Switching Rate')

    fig.suptitle(f'{video_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_detection_quality_bars(video_name, dq_data, out_path):
    """Bar chart for detection quality metrics."""
    policies = [p for p in POLICIES if p in dq_data]
    f1s = [dq_data[p].get("f1", 0) for p in policies]
    precs = [dq_data[p].get("precision", 0) for p in policies]
    recs = [dq_data[p].get("recall", 0) for p in policies]
    colors = [POLICY_COLORS.get(p, '#999') for p in policies]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(policies))
    w = 0.25

    ax.bar(x - w, precs, w, label='Precision', color='#3498DB', alpha=0.8)
    ax.bar(x, recs, w, label='Recall', color='#E74C3C', alpha=0.8)
    ax.bar(x + w, f1s, w, label='F1', color='#2ECC71', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)')
    ax.set_title(f'Detection Quality: {video_name}')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_grand_summary(grand_data, out_dir):
    """Grand summary plots across all 12 videos."""
    grand = grand_data["policies"]

    # 1. Pareto front scatter
    fig, ax = plt.subplots(figsize=(10, 7))
    for p in POLICIES:
        g = grand[p]
        t = g["weighted_T_total_mean"]
        s = g["weighted_slow_pct"]
        c = POLICY_COLORS.get(p, '#999')
        marker = 'o' if p in ["n_only", "s_only"] else 's'
        size = 150 if p in ["conf_ema", "combined_hyst"] else 80
        ax.scatter(t, s, c=c, s=size, marker=marker, zorder=5, edgecolors='black', linewidth=0.5)
        offset_x = 2 if p != "niqe_switch" else -15
        offset_y = 2
        ax.annotate(p, (t, s), fontsize=8, ha='left',
                    xytext=(offset_x, offset_y), textcoords='offset points')

    ax.set_xlabel('T_total mean (ms)', fontsize=12)
    ax.set_ylabel('s-route % (accuracy proxy)', fontsize=12)
    ax.set_title('Pareto Front: Latency vs Accuracy (12 Videos, 78,225 Frames)', fontsize=13)
    ax.grid(True, alpha=0.3)
    # Draw Pareto line through conf_ema and combined_hyst
    ce = grand["conf_ema"]
    ch = grand["combined_hyst"]
    ax.plot([grand["n_only"]["weighted_T_total_mean"], ce["weighted_T_total_mean"],
             ch["weighted_T_total_mean"], grand["s_only"]["weighted_T_total_mean"]],
            [0, ce["weighted_slow_pct"], ch["weighted_slow_pct"], 100],
            'k--', alpha=0.3, linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "grand_pareto_front.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: grand_pareto_front.png")

    # 2. CV comparison bar
    fig, ax = plt.subplots(figsize=(10, 5))
    policies_sorted = sorted(POLICIES, key=lambda p: grand[p]["cv_T_total_mean_pct"])
    cvs = [grand[p]["cv_T_total_mean_pct"] for p in policies_sorted]
    colors = [POLICY_COLORS.get(p, '#999') for p in policies_sorted]
    bars = ax.barh(policies_sorted, cvs, color=colors)
    ax.set_xlabel('CV of T_total mean (%)')
    ax.set_title('Cross-Video Consistency (Lower = Better)')
    for bar, cv in zip(bars, cvs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cv:.1f}%', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(out_dir / "grand_cv_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: grand_cv_comparison.png")

    # 3. Grand timing comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(POLICIES))
    t_means = [grand[p]["weighted_T_total_mean"] for p in POLICIES]
    t_p95s = [grand[p]["weighted_T_total_p95"] for p in POLICIES]
    colors = [POLICY_COLORS.get(p, '#999') for p in POLICIES]

    ax.bar(x - 0.2, t_means, 0.4, label='T_mean', color=colors, alpha=0.9)
    ax.bar(x + 0.2, t_p95s, 0.4, label='T_p95', color=colors, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(POLICIES, rotation=45, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Grand Timing: Mean vs P95 (12 Videos)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "grand_timing_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: grand_timing_bars.png")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Regenerating ALL plots from 12-video results")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    # Load grand summary
    with open(RESULTS_DIR / "overnight" / "grand_summary_12videos.json") as f:
        grand_data = json.load(f)

    # Grand summary plots
    print("\n[1/4] Grand summary plots...")
    plot_grand_summary(grand_data, FIGURES_DIR)

    # Per-video plots
    print("\n[2/4] Per-video comparison bars...")
    for material in ["glass", "porcelain"]:
        mat_dir = RESULTS_DIR / "overnight" / material
        if not mat_dir.exists():
            continue
        for vid_dir in sorted(mat_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            pc = vid_dir / "policy_comparison.json"
            if pc.exists():
                with open(pc) as f:
                    comp = json.load(f)
                out = FIGURES_DIR / f"comparison_{vid_dir.name}.png"
                plot_comparison_bars(vid_dir.name, comp, out)
                print(f"  Saved: comparison_{vid_dir.name}.png")

    # Detection quality plots
    print("\n[3/4] Detection quality bars...")
    dq_dir = RESULTS_DIR / "detection_quality"
    if dq_dir.exists():
        for vid_dir in sorted(dq_dir.iterdir()):
            if vid_dir.is_dir():
                dq_file = vid_dir / "detection_quality.json"
                if dq_file.exists():
                    with open(dq_file) as f:
                        dq = json.load(f)
                    out = FIGURES_DIR / f"detection_quality_{vid_dir.name}.png"
                    plot_detection_quality_bars(vid_dir.name, dq, out)
                    print(f"  Saved: detection_quality_{vid_dir.name}.png")

    # Timing traces (from CSVs)
    print("\n[4/4] Timing trace timeseries...")
    count = 0
    for material in ["glass", "porcelain"]:
        mat_dir = RESULTS_DIR / "overnight" / material
        if not mat_dir.exists():
            continue
        for vid_dir in sorted(mat_dir.iterdir()):
            if not vid_dir.is_dir():
                continue
            for policy in POLICIES:
                trace_csv = vid_dir / f"{policy}_trace.csv"
                if trace_csv.exists():
                    out = FIGURES_DIR / f"timeseries_{vid_dir.name}_{policy}.png"
                    plot_timing_timeseries(trace_csv, out,
                                           title=f"{vid_dir.name} / {policy}")
                    count += 1
    print(f"  Generated {count} timing trace plots")

    print(f"\n{'='*60}")
    print(f"ALL PLOTS REGENERATED: {FIGURES_DIR}")
    total = len(list(FIGURES_DIR.glob("*.png")))
    print(f"Total figures: {total}")
    print(f"{'='*60}")
