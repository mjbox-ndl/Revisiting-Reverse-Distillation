"""
Load inference results and collect by same class + same checkpoint_folder.
Draw 3 plots per class: auroc_sp, auroc_px, aupro_px (x=type, y=score).
Same class: multiple checkpoint_folders shown in one plot with different colors.
"""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Path to inference results (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "experiments" / "inference_results.json"

METRICS = ("auroc_sp", "auroc_px", "aupro_px")
METRIC_LABELS = {"auroc_sp": "AUROC (sample)", "auroc_px": "AUROC (pixel)", "aupro_px": "AUPRO (pixel)"}
# Colors for checkpoints (reused if more checkpoints than colors)
CHECKPOINT_COLORS = ["#2e86ab", "#e94f37", "#44af69", "#fcab10", "#7b2d8e"]


def load_results(path=None):
    path = path or RESULTS_PATH
    with open(path) as f:
        return json.load(f)


def collect_by_class_and_checkpoint(results):
    """
    Group results by (class, checkpoint_folder).
    Returns: dict[(class, checkpoint_folder)] -> list of result dicts
    """
    grouped = defaultdict(list)
    for r in results:
        key = (r["class"], r["checkpoint_folder"])
        grouped[key].append(r)
    return dict(grouped)


def build_class_plot_data(grouped):
    """
    For each class: list of checkpoints, list of types, and per-metric (type -> checkpoint -> score).
    Returns: dict[class] -> { "checkpoints": [...], "types": [...], "metrics": { metric: (types, ckpt_idx -> scores per type) } }
    """
    by_class = defaultdict(lambda: defaultdict(dict))  # class -> checkpoint -> type -> {metric: value}
    for (cls, ckpt), entries in grouped.items():
        for e in entries:
            t = e["types"][0] if e["types"] else "all"
            by_class[cls][ckpt][t] = {m: e[m] for m in METRICS}

    plot_data = {}
    for cls in by_class:
        checkpoints = sorted(by_class[cls].keys())
        all_types = set()
        for ckpt in checkpoints:
            all_types.update(by_class[cls][ckpt].keys())
        types = sorted(all_types, key=lambda x: (x == "all", x))

        metrics_data = {}
        for metric in METRICS:
            # for each type, list of scores in checkpoint order (NaN if missing)
            type_scores = []
            for t in types:
                row = []
                for ckpt in checkpoints:
                    row.append(by_class[cls][ckpt].get(t, {}).get(metric, np.nan))
                type_scores.append(row)
            metrics_data[metric] = type_scores  # list of [scores per checkpoint] per type

        plot_data[cls] = {"checkpoints": checkpoints, "types": types, "metrics": metrics_data}
    return plot_data


def draw_plots_per_class(plot_data):
    """One figure per class with 3 subplots. Each subplot: x=type, grouped bars per checkpoint (different colors)."""
    for cls, data in sorted(plot_data.items()):
        checkpoints = data["checkpoints"]
        types = data["types"]
        n_types = len(types)
        n_ckpt = len(checkpoints)
        colors = [CHECKPOINT_COLORS[i % len(CHECKPOINT_COLORS)] for i in range(n_ckpt)]

        x = np.arange(n_types)
        total_width = 0.8
        bar_width = total_width / n_ckpt
        offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_ckpt)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(f"Class: {cls}", fontsize=14)

        for ax, metric in zip(axes, METRICS):
            type_scores = data["metrics"][metric]  # list of length n_types, each element list of n_ckpt scores
            for ckpt_idx in range(n_ckpt):
                scores = [type_scores[i][ckpt_idx] for i in range(n_types)]
                pos = x + offsets[ckpt_idx]
                label = Path(checkpoints[ckpt_idx]).name or checkpoints[ckpt_idx]
                ax.bar(pos, scores, width=bar_width, label=label, color=colors[ckpt_idx], edgecolor="black", linewidth=0.5, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(types, rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.set_xlabel("Type")
            ax.set_title(METRIC_LABELS[metric])
            ax.set_ylim(0.85, 1.025)
            ax.legend(loc="lower right", fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out_path = SCRIPT_DIR / "experiments" / f"results_{cls}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


def main():
    results = load_results()
    grouped = collect_by_class_and_checkpoint(results)

    for (cls, ckpt), entries in sorted(grouped.items()):
        print(f"\n{cls} | {ckpt} ({len(entries)} runs)")
        for e in entries:
            types_str = ", ".join(e["types"])
            print(f"  types: [{types_str}]  auroc_sp={e['auroc_sp']:.4f}  auroc_px={e['auroc_px']:.4f}  aupro_px={e['aupro_px']:.4f}")

    plot_data = build_class_plot_data(grouped)
    draw_plots_per_class(plot_data)

    return grouped


if __name__ == "__main__":
    main()
