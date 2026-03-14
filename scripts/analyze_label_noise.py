"""Generate plots for Label Noise Ablation experiments.

This script reads the summary CSVs produced by `run_label_noise_ablation` and
produces visualization artifacts showing how test accuracy / loss degrade as label
noise increases.

Outputs:
  - test_accuracy_vs_noise.png
  - test_loss_vs_noise.png
  - retention_rate_vs_noise.png
  - label_noise_summary_with_retention.csv

Run:
  python scripts/analyze_label_noise.py
"""

from pathlib import Path
import sys

# Ensure the repository root is on sys.path for `src` imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def _load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def compute_retention(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute retention (relative test accuracy) vs noise rate per optimizer."""
    out_rows = []
    for optimizer, group in summary_df.groupby("optimizer"):
        # baseline at noise_rate == 0.0
        baseline = group[group["noise_rate"] == 0.0]
        if baseline.empty:
            continue
        baseline_acc = float(baseline["test_acc_mean"].iloc[0])
        for _, row in group.iterrows():
            noise_rate = float(row["noise_rate"])
            test_acc = float(row["test_acc_mean"])
            retention = (test_acc / baseline_acc) * 100.0 if baseline_acc > 0 else float("nan")
            out_rows.append({
                "optimizer": optimizer,
                "noise_rate": noise_rate,
                "test_acc_mean": test_acc,
                "test_acc_std": float(row.get("test_acc_std", float("nan"))),
                "baseline_acc": baseline_acc,
                "retention_pct": retention,
            })
    return pd.DataFrame(out_rows)


def compute_accuracy_drop(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute accuracy drop (clean - noisy) vs noise rate per optimizer."""
    out_rows = []
    for optimizer, group in summary_df.groupby("optimizer"):
        baseline = group[group["noise_rate"] == 0.0]
        if baseline.empty:
            continue
        baseline_acc = float(baseline["test_acc_mean"].iloc[0])
        for _, row in group.iterrows():
            noise_rate = float(row["noise_rate"])
            test_acc = float(row["test_acc_mean"])
            drop = baseline_acc - test_acc
            out_rows.append({
                "optimizer": optimizer,
                "noise_rate": noise_rate,
                "test_acc_mean": test_acc,
                "test_acc_std": float(row.get("test_acc_std", float("nan"))),
                "baseline_acc": baseline_acc,
                "accuracy_drop": drop,
            })
    return pd.DataFrame(out_rows)


def plot_label_noise_summary(
    summary_df: pd.DataFrame,
    out_dir: Path,
    title_suffix: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure sorted noise rates
    summary_df = summary_df.sort_values(["optimizer", "noise_rate"])

    # Plot 1: Test Accuracy vs Noise Rate (mean ± std)
    fig, ax = plt.subplots(figsize=(8, 5))
    for optimizer, group in summary_df.groupby("optimizer"):
        ax.errorbar(
            group["noise_rate"],
            group["test_acc_mean"],
            yerr=group["test_acc_std"],
            marker="o",
            linestyle="-",
            capsize=4,
            label=optimizer,
        )

    ax.set_xlabel("Label Noise Rate")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Test Accuracy vs Label Noise {title_suffix}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize="small")
    plt.tight_layout()
    fig_path = out_dir / f"label_noise_test_accuracy_{title_suffix.strip().replace(' ', '_')}.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Plot 2: Test Loss vs Noise Rate
    if "test_loss_mean" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for optimizer, group in summary_df.groupby("optimizer"):
            ax.errorbar(
                group["noise_rate"],
                group["test_loss_mean"],
                yerr=group.get("test_loss_std"),
                marker="o",
                linestyle="-",
                capsize=4,
                label=optimizer,
            )

        ax.set_xlabel("Label Noise Rate")
        ax.set_ylabel("Test Loss")
        ax.set_title(f"Test Loss vs Label Noise {title_suffix}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        fig_path = out_dir / f"label_noise_test_loss_{title_suffix.strip().replace(' ', '_')}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    # Plot 3: Accuracy Drop (clean - noisy) vs Noise Rate
    drop_df = compute_accuracy_drop(summary_df)
    if not drop_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for optimizer, group in drop_df.groupby("optimizer"):
            ax.plot(
                group["noise_rate"],
                group["accuracy_drop"],
                marker="o",
                linestyle="-",
                label=optimizer,
            )

        ax.set_xlabel("Label Noise Rate")
        ax.set_ylabel("Accuracy Drop (clean - noisy)")
        ax.set_title(f"Accuracy Drop vs Label Noise {title_suffix}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize="small")
        plt.tight_layout()
        fig_path = out_dir / f"label_noise_accuracy_drop_{title_suffix.strip().replace(' ', '_')}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

        # Save drop table for inspection
        drop_df.to_csv(out_dir / f"label_noise_accuracy_drop_{title_suffix.strip().replace(' ', '_')}.csv", index=False)

    # Return for additional analysis
    return summary_df


def main() -> None:
    root = Path("f:/GDSearch/results_proposal_full_20260223_v2")
    out_dir = root / "analysis" / "label_noise"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Support both the "results" CSV (raw per-epoch/seed results) and a potential
    # "summary" CSV (already aggregated). Many runs currently only write the
    # raw results file, so this helper will aggregate if needed.
    def _load_or_compute_summary(results_csv: Path) -> pd.DataFrame:
        if not results_csv.exists():
            raise FileNotFoundError(f"Missing label-noise results CSV: {results_csv}")
        df = pd.read_csv(results_csv)
        # If a summary already exists (already aggregated), just return it
        required_cols = {"optimizer", "noise_rate", "test_acc_mean"}
        if required_cols.issubset(set(df.columns)):
            return df

        # Otherwise, compute summary in the same way as create_label_noise_summary
        final_results = df.groupby(["optimizer", "noise_rate", "seed"]).last().reset_index()
        summary = final_results.groupby(["optimizer", "noise_rate"]).agg({
            "train_acc": ["mean", "std"],
            "val_acc": ["mean", "std"],
            "test_acc": ["mean", "std"],
            "train_loss": ["mean", "std"],
            "val_loss": ["mean", "std"],
            "test_loss": ["mean", "std"],
        }).reset_index()
        summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
        return summary

    # Plot MNIST MLP
    mnist_res_csv = root / "experiments" / "label_noise" / "label_noise_results_mnist_mlp.csv"
    if mnist_res_csv.exists():
        mnist_df = _load_or_compute_summary(mnist_res_csv)
        mnist_df = mnist_df.sort_values(["optimizer", "noise_rate"])
        plot_label_noise_summary(mnist_df, out_dir, "(MNIST MLP)")

        mnist_ret = compute_retention(mnist_df)
        mnist_ret.to_csv(out_dir / "label_noise_mnist_mlp_retention.csv", index=False)

    # Plot CIFAR-10 ResNet18
    cifar_res_csv = root / "experiments" / "label_noise" / "label_noise_results_cifar10_resnet18.csv"
    if cifar_res_csv.exists():
        cifar_df = _load_or_compute_summary(cifar_res_csv)
        cifar_df = cifar_df.sort_values(["optimizer", "noise_rate"])
        plot_label_noise_summary(cifar_df, out_dir, "(CIFAR-10 ResNet18)")

        cifar_ret = compute_retention(cifar_df)
        cifar_ret.to_csv(out_dir / "label_noise_cifar10_resnet18_retention.csv", index=False)

    print("Label noise plots and retention tables written to:", out_dir)


if __name__ == "__main__":
    main()
