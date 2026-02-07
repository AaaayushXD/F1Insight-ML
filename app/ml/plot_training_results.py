"""
Plot F1Insight training results from evaluation_report.json.
Generates matplotlib figures and saves to app/ml/outputs/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.json"

# Feature names for importance plot (must match train.py order: numeric then categorical)
FEATURE_NAMES = [
    "qualifying_position",
    "grid_position",
    "driver_prior_points",
    "driver_prior_wins",
    "driver_prior_position",
    "constructor_prior_points",
    "constructor_prior_wins",
    "constructor_prior_position",
    "total_stops",
    "mean_stop_duration",
    "circuit_lat",
    "circuit_lng",
    "driver_id",
    "constructor_id",
    "circuit_id",
]


def load_report(path: Path = None) -> dict:
    path = path or REPORT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}. Run training first.")
    with open(path) as f:
        return json.load(f)


def plot_regression_metrics(report: dict, ax: plt.Axes) -> None:
    reg = report.get("regression", {})
    if not reg:
        ax.text(0.5, 0.5, "No regression metrics", ha="center", va="center")
        return
    models = list(reg.keys())
    x = np.arange(len(models))
    width = 0.35

    train_mae = [reg[m].get("train_mae", np.nan) for m in models]
    train_rmse = [reg[m].get("train_rmse", np.nan) for m in models]

    bars1 = ax.bar(x - width / 2, train_mae, width, label="MAE (train)", color="#2C64DD", alpha=0.9)
    bars2 = ax.bar(x + width / 2, train_rmse, width, label="RMSE (train)", color="#A1BAF0", alpha=0.9)

    ax.set_ylabel("Error (positions)")
    ax.set_title("Regression: Finish position")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=15, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    y_max = max(max(train_mae), max(train_rmse)) * 1.15 if train_mae else 1
    ax.set_ylim(0, y_max)
    for b in bars1:
        ax.annotate(f"{b.get_height():.2f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=0)
    for b in bars2:
        ax.annotate(f"{b.get_height():.2f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=0)


def plot_classification_metrics(report: dict, ax: plt.Axes) -> None:
    cls = report.get("classification", {})
    podium = cls.get("is_podium", {})
    if not podium or "status" in podium:
        ax.text(0.5, 0.5, "No podium classification metrics", ha="center", va="center")
        return
    models = [k for k in podium if isinstance(podium[k], dict) and "train_accuracy" in podium[k]]
    if not models:
        ax.text(0.5, 0.5, "No models", ha="center", va="center")
        return
    x = np.arange(len(models))
    width = 0.35

    acc = [podium[m]["train_accuracy"] * 100 for m in models]
    f1 = [podium[m].get("train_f1", 0) * 100 for m in models]

    bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy % (train)", color="#2C64DD", alpha=0.9)
    bars2 = ax.bar(x + width / 2, f1, width, label="F1 % (train)", color="#A1BAF0", alpha=0.9)

    ax.set_ylabel("Score (%)")
    ax.set_title("Classification: Podium (top 3)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], rotation=15, ha="right")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)
    for b in bars1:
        ax.annotate(f"{b.get_height():.0f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for b in bars2:
        ax.annotate(f"{b.get_height():.0f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)


def plot_feature_importance(report: dict, ax: plt.Axes, model_key: str = "random_forest") -> None:
    reg = report.get("regression", {})
    model_data = reg.get(model_key, {})
    imp = model_data.get("feature_importance")
    if not imp:
        ax.text(0.5, 0.5, f"No feature importance for {model_key}", ha="center", va="center")
        return
    indices = sorted([int(k[1:]) for k in imp.keys()])
    names = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i}" for i in indices]
    values = [imp[f"f{i}"] for i in indices]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(values)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance ({model_key.replace('_', ' ').title()})")


def plot_temporal_split(report: dict, ax: plt.Axes) -> None:
    split = report.get("temporal_split", {})
    if not split:
        ax.axis("off")
        return
    text = "Temporal split\n" + "\n".join(f"  {k}: {v}" for k, v in split.items())
    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11, verticalalignment="center",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="#E8EEFF", alpha=0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Data split")


def main(save_path: Path = None, report_path: Path = None) -> Path:
    report = load_report(report_path)
    save_path = save_path or OUTPUT_DIR / "training_results.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("F1Insight — Training results", fontsize=14, fontweight="bold")

    plot_temporal_split(report, axes[0, 0])
    plot_regression_metrics(report, axes[0, 1])
    plot_classification_metrics(report, axes[1, 0])
    plot_feature_importance(report, axes[1, 1], model_key="random_forest")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_all_in_one(save_dir: Path = None) -> None:
    """Create one combined figure and one feature-importance-only figure."""
    report = load_report()
    save_dir = save_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("F1Insight — Training results", fontsize=14, fontweight="bold")
    plot_temporal_split(report, axes[0, 0])
    plot_regression_metrics(report, axes[0, 1])
    plot_classification_metrics(report, axes[1, 0])
    plot_feature_importance(report, axes[1, 1], model_key="random_forest")
    plt.tight_layout()
    path1 = save_dir / "training_results.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path1}")

    # XGBoost feature importance (often different from RF)
    reg = report.get("regression", {})
    if "xgboost" in reg and reg["xgboost"].get("feature_importance"):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_feature_importance(report, ax2, model_key="xgboost")
        plt.tight_layout()
        path2 = save_dir / "feature_importance_xgboost.png"
        plt.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {path2}")


if __name__ == "__main__":
    plot_all_in_one()
