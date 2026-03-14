"""Quick label-noise ablation run (small sweeps) for plotting.

This script runs a small subset of the full label-noise experiment (few seeds + few epochs)
so that we can generate plots showing multiple noise rates.
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so we can import `src` modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.run_label_noise_ablation import run_label_noise_ablation, LabelNoiseConfig


def main():
    config = LabelNoiseConfig(
        noise_rates=[0.0, 0.1, 0.2, 0.4],
        seeds=[42, 123],
        epochs=3,
        batch_size=128,
        num_workers=0,  # Avoid shared memory/DataLoader worker issues on Windows
    )

    optimizers = {
        'SGD': {'lr': 0.01, 'momentum': 0.0},
        'SGD_Momentum': {'lr': 0.01, 'momentum': 0.9},
        'Adam': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'AdamW': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'weight_decay': 0.01},
    }

    out_dir = 'results/label_noise_quick_for_plot'

    print('Running quick label-noise ablation (CIFAR-10 ResNet18) ...')
    run_label_noise_ablation(
        dataset_name='cifar10',
        model_name='resnet18',
        optimizers_config=optimizers,
        config=config,
        output_dir=out_dir,
    )
    print('Done. Results saved to:', out_dir)


if __name__ == '__main__':
    main()
