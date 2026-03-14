from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path("f:/GDSearch/results_longrun_20260314/experiments/experiments/batch_ablation/MNIST_batch_ablation_seeds42_123_456_789.csv")
    out_dir = Path("f:/GDSearch/results_longrun_20260314/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    per_opt = (
        df.groupby(["optimizer", "batch_size", "scaled_lr"], as_index=False)
        .agg(
            acc_mean=("final_test_acc", "mean"),
            acc_std=("final_test_acc", "std"),
            loss_mean=("final_loss", "mean"),
            loss_std=("final_loss", "std"),
            n=("seed", "count"),
        )
    )

    per_batch = (
        df.groupby(["batch_size", "scaled_lr"], as_index=False)
        .agg(
            acc_mean=("final_test_acc", "mean"),
            acc_std=("final_test_acc", "std"),
            loss_mean=("final_loss", "mean"),
            loss_std=("final_loss", "std"),
            n=("seed", "count"),
        )
    )

    # Ranking score: high acc, low variance, low loss
    per_opt["score"] = per_opt["acc_mean"] - 0.6 * per_opt["acc_std"] - 8.0 * per_opt["loss_mean"]
    per_batch["score"] = per_batch["acc_mean"] - 0.6 * per_batch["acc_std"] - 8.0 * per_batch["loss_mean"]

    per_opt_sorted = per_opt.sort_values(["optimizer", "score"], ascending=[True, False])
    per_batch_sorted = per_batch.sort_values("score", ascending=False)

    per_opt_sorted.to_csv(out_dir / "batch_ablation_per_optimizer_stats.csv", index=False)
    per_batch_sorted.to_csv(out_dir / "batch_ablation_per_batchsize_stats.csv", index=False)

    # Combined visualization (4 optimizers together)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    colors = {
        "SGD": "#1f77b4",
        "SGD_Momentum": "#ff7f0e",
        "Adam": "#2ca02c",
        "AdamW": "#d62728",
    }

    for optimizer, group in per_opt_sorted.groupby("optimizer"):
        group = group.sort_values("batch_size")
        c = colors.get(optimizer, None)
        ax1.errorbar(group["batch_size"], group["acc_mean"], yerr=group["acc_std"], marker="o", linewidth=2, capsize=4, label=optimizer, color=c)
        ax2.errorbar(group["batch_size"], group["loss_mean"], yerr=group["loss_std"], marker="o", linewidth=2, capsize=4, label=optimizer, color=c)

    ax1.set_title("Final Test Accuracy vs Batch Size (mean±std)")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_title("Final Loss vs Batch Size (mean±std)")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    pb = per_batch_sorted.sort_values("batch_size")
    ax3.errorbar(pb["batch_size"], pb["acc_mean"], yerr=pb["acc_std"], marker="o", linewidth=2, capsize=4, color="#6a3d9a")
    ax3.set_title("Overall Accuracy by Batch Size (all optimizers)")
    ax3.set_xlabel("Batch size")
    ax3.set_ylabel("Accuracy (%)")
    ax3.grid(alpha=0.3)

    ax4.errorbar(pb["batch_size"], pb["loss_mean"], yerr=pb["loss_std"], marker="o", linewidth=2, capsize=4, color="#b15928")
    ax4.set_title("Overall Loss by Batch Size (all optimizers)")
    ax4.set_xlabel("Batch size")
    ax4.set_ylabel("Loss")
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "batch_ablation_combined_4optimizers.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    best_batch = per_batch_sorted.iloc[0]
    worst_batch = per_batch_sorted.iloc[-1]

    best_per_optimizer = per_opt_sorted.groupby("optimizer", as_index=False).first()

    lines = []
    lines.append("# Batch Ablation Combined Report (4 Optimizers)\n")
    lines.append("## Overall Batch Size Recommendation\n")
    lines.append(
        f"- Batch size ổn nhất (overall): **{int(best_batch['batch_size'])}** "
        f"(scaled_lr={best_batch['scaled_lr']}, acc={best_batch['acc_mean']:.3f}±{best_batch['acc_std']:.3f}, "
        f"loss={best_batch['loss_mean']:.4f}±{best_batch['loss_std']:.4f})"
    )
    lines.append(
        f"- Batch size kém nhất (overall): **{int(worst_batch['batch_size'])}** "
        f"(scaled_lr={worst_batch['scaled_lr']}, acc={worst_batch['acc_mean']:.3f}±{worst_batch['acc_std']:.3f}, "
        f"loss={worst_batch['loss_mean']:.4f}±{worst_batch['loss_std']:.4f})"
    )

    lines.append("\n## Best per Optimizer\n")
    for _, r in best_per_optimizer.iterrows():
        lines.append(
            f"- {r['optimizer']}: batch={int(r['batch_size'])}, lr={r['scaled_lr']}, "
            f"acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}, loss={r['loss_mean']:.4f}±{r['loss_std']:.4f}"
        )

    lines.append("\n## Learning Rate Mapping (Linear Scaling in this ablation)\n")
    lines.append("- batch=32 -> lr=0.00125")
    lines.append("- batch=256 -> lr=0.01")
    lines.append("- batch=512 -> lr=0.02")

    (out_dir / "batch_ablation_combined_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("Wrote:", out_dir / "batch_ablation_combined_4optimizers.png")
    print("Wrote:", out_dir / "batch_ablation_per_optimizer_stats.csv")
    print("Wrote:", out_dir / "batch_ablation_per_batchsize_stats.csv")
    print("Wrote:", out_dir / "batch_ablation_combined_report.md")


if __name__ == "__main__":
    main()
