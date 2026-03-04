"""Run quick 2D beta-sensitivity demos and save artifacts under a chosen root."""

import argparse
import subprocess
import sys
from pathlib import Path


def _cleanup_demo_outputs(output_root: Path) -> None:
    """
    Remove previous demo artifacts so regenerated plots/CSVs are always fresh.
    """
    targets = [
        output_root / "rosenbrock" / "momentum",
        output_root / "saddle_point" / "adam",
    ]
    for d in targets:
        if not d.exists():
            continue
        for p in d.glob("*"):
            if p.is_file() and p.suffix.lower() in {".png", ".csv", ".npy"}:
                try:
                    p.unlink()
                except Exception:
                    pass


def run_beta_sensitivity_2d_demos(output_root: Path) -> int:
    """Run momentum/adam beta-sensitivity demos for 2D functions."""
    output_root = Path(output_root)
    _cleanup_demo_outputs(output_root)

    print("=" * 80)
    print("BETA SENSITIVITY 2D VISUALIZATIONS")
    print("=" * 80)
    print("Running quick visualizations for thesis figures...")
    print(f"Output root: {output_root}")
    print()

    demos = [
        {
            "name": "Momentum beta Sweep on Rosenbrock",
            "args": [
                "--optimizer",
                "Momentum",
                "--function",
                "rosenbrock",
                "--beta-values",
                "0.5,0.7,0.9,0.95,0.99",
                "--lr",
                "0.001",
                "--max-iters",
                "1500",
                "--output-dir",
                str(output_root),
            ],
            "description": "Shows beta impact on trajectory smoothness and convergence speed",
        },
        {
            "name": "Adam beta1 x beta2 on Saddle Point",
            "args": [
                "--optimizer",
                "Adam",
                "--function",
                "saddle_point",
                "--beta1-values",
                "0.8,0.9",
                "--beta2-values",
                "0.9,0.99",
                "--max-iters",
                "200",
                "--output-dir",
                str(output_root),
            ],
            "description": "Demonstrates Adam saddle-point escape dynamics",
        },
    ]

    failures = 0
    for demo in demos:
        print(f"\n{'=' * 80}")
        print(f"Running: {demo['name']}")
        print(f"Description: {demo['description']}")
        print(f"{'=' * 80}\n")

        cmd = [sys.executable, "src/experiments/beta_sensitivity_2d.py"] + demo["args"]
        try:
            subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n[OK] {demo['name']} completed successfully")
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"\n[ERROR] {demo['name']} failed with code {e.returncode}")
            print("Continuing with remaining demonstrations...")
        except Exception as e:
            failures += 1
            print(f"\n[ERROR] Unexpected error: {e}")

    print("\n" + "=" * 80)
    print("BETA SENSITIVITY 2D DEMONSTRATIONS COMPLETE")
    print("=" * 80)
    print("\nGenerated visualizations can be found in:")
    print(f"  - {output_root / 'rosenbrock' / 'momentum'}")
    print(f"  - {output_root / 'saddle_point' / 'adam'}")
    print("=" * 80)

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 2D beta-sensitivity demo plots")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/beta_sensitivity_2d",
        help="Root output directory for beta-sensitivity demo artifacts",
    )
    args = parser.parse_args()
    return run_beta_sensitivity_2d_demos(Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
