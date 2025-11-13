# Quick Start: MNIST Publication Experiments on Kaggle

Follow these steps to run the MNIST experiment suite in a Kaggle Notebook.

1) Open Kaggle Notebooks
- Go to https://www.kaggle.com/code and click "New Notebook"

2) Configure notebook settings
- Accelerator: GPU (T4 or P100)
- Internet: ON (required to download MNIST from torchvision)

3) Add files
- Upload these two files into the notebook session (left sidebar → Add data → Upload):
  - `mnist_publication.py`
  - (optional) `requirements.txt` — only if you need to install packages not already preinstalled

4) Install dependencies (optional)
- Kaggle normally has torch, torchvision, numpy, pandas, scipy, tqdm preinstalled.
- If needed, run:

```python
!pip -q install -r requirements.txt
```

5) Run the experiments
- In a code cell, run:

```python
!python mnist_publication.py --seeds 1,2,3,4,5,6,7,8,9,10 --epochs 10 --results-dir results
```

Flags:
- `--seeds`: comma-separated seed list (default: 1-10)
- `--epochs`: training epochs per run (default: 10)
- `--batch-size`: batch size (default: 128)
- `--quick`: run fewer seeds/epochs quickly (overrides seeds to 1-3 and epochs to 3)

6) Outputs
- Files appear in `/kaggle/working/results`:
  - Per-run CSVs: `NN_SimpleMLP_MNIST_<Optimizer>_lr<lr>_seed<seed>_publication.csv`
  - Statistical comparison CSV: `mnist_statistical_comparisons_publication.csv`

7) Shorten runtime (if needed)
- Use `--quick` to run 3 seeds × 5 optimizers = 15 runs, 3 epochs each
- Or reduce `--seeds` or `--epochs`

8) After running
- Download the `results/` folder as a zip for your paper
- Share the statistical CSV and plots as needed
