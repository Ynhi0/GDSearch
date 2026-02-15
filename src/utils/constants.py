"""
Project-wide constants and magic numbers with documentation.

This module eliminates magic numbers scattered throughout the codebase
by providing named constants with clear documentation of their purpose.

Principle: "Replace magic numbers with named constants that explain WHY"

Example:
    >>> from src.utils.constants import MAX_SAFE_LOSS, DEFAULT_BATCH_SIZE_MNIST
    >>> if loss > MAX_SAFE_LOSS:
    ...     logging.error("Numerical instability detected!")
"""

# ============================================================================
# NUMERICAL STABILITY THRESHOLDS
# ============================================================================

MAX_SAFE_LOSS = 1e10
"""
Maximum safe loss value before numerical instability.

Losses exceeding this threshold indicate gradient explosion or numerical
overflow. Training should be stopped or batch size reduced.
"""

MIN_SAFE_LOSS = 1e-10
"""
Minimum safe loss value (prevents log(0) errors).

Used as epsilon in logarithmic computations to avoid -inf values.
"""

GRADIENT_EXPLOSION_THRESHOLD = 1e6
"""
Gradient norm threshold for explosion detection.

If gradient norm exceeds this, the model is experiencing gradient explosion
and gradient clipping or learning rate reduction is needed.
"""

EPSILON = 1e-8
"""
Small constant for numerical stability (division, logarithms).

Standard epsilon value used in Adam, layer normalization, etc.
"""

# ============================================================================
# DEFAULT BATCH SIZES (Optimized for T4 GPU with 15GB VRAM)
# ============================================================================

DEFAULT_BATCH_SIZE_MNIST = 128
"""
Default batch size for MNIST experiments.

Optimized for T4 GPU (15GB VRAM). MNIST has small images (28x28x1),
so larger batches are feasible without OOM.
"""

DEFAULT_BATCH_SIZE_CIFAR10 = 128
"""
Default batch size for CIFAR-10 experiments.

Optimized for T4 GPU. CIFAR-10 images (32x32x3) are small enough
for batch size of 128 with ResNet-18.
"""

DEFAULT_BATCH_SIZE_NLP = 32
"""
Default batch size for NLP/Transformer experiments.

Transformer models have large memory footprint due to attention mechanism.
Batch size 32 is safe for most transformer models on T4 GPU.
"""

DEFAULT_BATCH_SIZE_MEDICAL = 16
"""
Default batch size for medical segmentation (U-Net).

Medical images are often high-resolution (128x128 or larger) with
segmentation masks. Batch size 16 is safe for U-Net on T4 GPU.
"""

# ============================================================================
# DEFAULT LEARNING RATES (Per-Optimizer Fair Defaults)
# ============================================================================

ADAM_DEFAULT_LR = 1e-3
"""
Standard default learning rate for Adam optimizer.

Based on original Adam paper (Kingma & Ba, 2015).
This is the canonical default for adaptive methods.
"""

ADAMW_DEFAULT_LR = 1e-3
"""
Standard default learning rate for AdamW optimizer.

AdamW uses same defaults as Adam for lr, betas, and eps.
Only weight_decay handling differs.
"""

SGD_DEFAULT_LR = 0.1
"""
Standard default learning rate for SGD optimizer.

SGD requires higher learning rates than adaptive methods.
0.1 is the canonical default for SGD on image classification.
"""

SGD_MOMENTUM_DEFAULT = 0.9
"""
Standard momentum coefficient for SGD with momentum.

0.9 is the most common momentum value in literature and provides
good balance between stability and acceleration.
"""

RMSPROP_DEFAULT_LR = 1e-2
"""
Standard default learning rate for RMSprop optimizer.

RMSprop typically uses lr=0.01 for image classification tasks.
"""

ADAGRAD_DEFAULT_LR = 1e-2
"""
Standard default learning rate for Adagrad optimizer.

Adagrad accumulates squared gradients, so higher initial lr is appropriate.
"""

# ============================================================================
# OPTIMIZER HYPERPARAMETER DEFAULTS
# ============================================================================

ADAM_BETA1 = 0.9
"""
Adam beta1 (exponential decay for first moment).

Controls how much history to retain for gradient mean estimation.
"""

ADAM_BETA2 = 0.999
"""
Adam beta2 (exponential decay for second moment).

Controls how much history to retain for gradient variance estimation.
"""

ADAM_EPSILON = 1e-8
"""
Adam epsilon for numerical stability.

Prevents division by zero in adaptive learning rate computation.
"""

ADAMW_WEIGHT_DECAY = 1e-2
"""
Default weight decay for AdamW.

AdamW paper recommends 0.01 for most tasks. This is applied
as true L2 regularization (not coupled with gradient updates).
"""

# ============================================================================
# TRAINING CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_EPOCHS_MNIST = 50
"""
Default training epochs for MNIST.

MNIST converges quickly; 50 epochs is sufficient for most optimizers
to reach near-optimal performance.
"""

DEFAULT_EPOCHS_CIFAR10 = 100
"""
Default training epochs for CIFAR-10.

CIFAR-10 requires more epochs than MNIST due to higher complexity.
100 epochs is standard for fair optimizer comparison.
"""

DEFAULT_EPOCHS_NLP = 10
"""
Default training epochs for NLP tasks.

Transformer models converge relatively quickly. 10 epochs is typical
for fine-tuning pre-trained models.
"""

DEFAULT_EPOCHS_MEDICAL = 50
"""
Default training epochs for medical segmentation.

Medical segmentation requires sufficient epochs for the model to learn
complex anatomical structures.
"""

DEFAULT_PATIENCE = 10
"""
Default patience for early stopping.

Number of epochs to wait for validation improvement before stopping.
10 epochs provides good balance between preventing overfitting and
allowing sufficient training time.
"""

DEFAULT_VAL_SPLIT = 0.15
"""
Default validation split ratio.

15% validation split is common practice, providing sufficient validation
samples while preserving most data for training.
"""

# ============================================================================
# GRADIENT CLIPPING THRESHOLDS
# ============================================================================

GRADIENT_CLIP_NORM_DEFAULT = 1.0
"""
Default gradient clipping norm.

Clips gradients to this L2 norm to prevent gradient explosion.
1.0 is a common default that prevents extreme gradients without
overly constraining gradient magnitudes.
"""

GRADIENT_CLIP_NORM_TRANSFORMERS = 1.0
"""
Gradient clipping norm for transformer models.

Transformers are prone to gradient explosion due to deep architecture
and attention mechanism. Clipping at 1.0 is standard practice.
"""

# ============================================================================
# OOM (Out of Memory) RECOVERY PARAMETERS
# ============================================================================

OOM_MIN_BATCH_SIZE = 1
"""
Minimum batch size for OOM recovery.

When OOM occurs, batch size is halved until this minimum is reached.
Batch size 1 is the last resort for training continuation.
"""

OOM_MAX_RETRIES = 3
"""
Maximum retry attempts for OOM recovery.

After 3 failed attempts to reduce batch size, the training is aborted
as the model is too large for available GPU memory.
"""

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================

CHECKPOINT_SAVE_INTERVAL = 1
"""
Checkpoint save interval (epochs).

Save checkpoint every N epochs. Value of 1 saves every epoch,
providing fine-grained recovery points.
"""

CHECKPOINT_KEEP_LAST_N = 3
"""
Number of recent checkpoints to keep.

To save disk space, only keep the N most recent checkpoints.
Set to -1 to keep all checkpoints.
"""

# ============================================================================
# SANITY CHECK THRESHOLDS
# ============================================================================

MIN_TRAIN_ACC_MNIST = 10.0
"""
Minimum expected training accuracy for MNIST (sanity check).

MNIST has 10 classes, so random guessing gives ~10% accuracy.
If training accuracy is below this after a few epochs, something is wrong
(e.g., only processing last batch instead of accumulating correctly).
"""

MIN_TRAIN_ACC_CIFAR10 = 10.0
"""
Minimum expected training accuracy for CIFAR-10 (sanity check).

CIFAR-10 has 10 classes. Training accuracy below 10% after epoch 2
indicates a critical bug in the training loop.
"""

# ============================================================================
# RANDOM SEED DEFAULTS
# ============================================================================

DEFAULT_RANDOM_SEED = 42
"""
Default random seed for reproducibility.

42 is the answer to life, the universe, and everything.
Also happens to be a good seed for most experiments.
"""

MULTI_SEED_DEFAULTS = [42, 123, 456]
"""
Default seeds for multi-seed experiments.

Three seeds provide reasonable statistical confidence for comparing
optimizer performance while keeping computational cost manageable.
"""

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

DEFAULT_NUM_WORKERS = 2
"""
Default number of dataloader worker processes.

2 workers provide good balance between data loading speed and
memory overhead. Increase for large datasets on multi-core CPUs.
"""

DEFAULT_PIN_MEMORY = True
"""
Default setting for pinning memory in dataloader.

Pinned memory improves data transfer speed from CPU to GPU
with minimal memory overhead. Recommended for CUDA training.
"""

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

MLFLOW_TRACKING_URI_DEFAULT = "mlruns/"
"""
Default MLflow tracking URI (local directory).

Experiments are logged to local 'mlruns' directory by default.
Override with environment variable MLFLOW_TRACKING_URI for remote tracking.
"""

LOG_INTERVAL_DEFAULT = 10
"""
Default logging interval (epochs).

Log training metrics every N epochs to reduce logging overhead
while maintaining visibility into training progress.
"""

# ============================================================================
# QUICK MODE PARAMETERS (for CI/testing)
# ============================================================================

QUICK_MODE_EPOCHS = 3
"""
Number of epochs for quick/smoke testing.

Used with --quick or --ultra-quick flags for rapid validation
that code runs without errors.
"""

ULTRA_QUICK_MODE_EPOCHS = 1
"""
Number of epochs for ultra-quick testing (minimal smoke test).

Single epoch for fastest possible validation in CI pipelines.
"""

QUICK_MODE_TRIALS = 3
"""
Number of hyperparameter tuning trials in quick mode.

Reduced from typical 10-20 trials for fast testing.
"""

# ============================================================================
# FILE NAMING CONVENTIONS
# ============================================================================

RESULTS_FILENAME_PATTERN = "NN_{model}_{dataset}_{optimizer}_lr{lr}_seed{seed}"
"""
Standard filename pattern for experiment results.

Format: NN_<Model>_<Dataset>_<Optimizer>_lr<lr>_seed<seed>.csv
Example: NN_ResNet18_CIFAR10_Adam_lr0.001_seed42.csv

This pattern is expected by plotting and analysis scripts.
"""

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_DPI = 300
"""
Default DPI for saved plots.

300 DPI is publication-quality resolution suitable for papers.
"""

PLOT_FIGSIZE_DEFAULT = (10, 6)
"""
Default figure size for plots (width, height in inches).

10x6 inches provides good aspect ratio for most plots.
"""

PLOT_STYLE = 'seaborn-v0_8-darkgrid'
"""
Default matplotlib style for plots.

Seaborn darkgrid provides professional-looking plots with gridlines
for easier value reading.
"""

# ============================================================================
# CONVERGENCE DETECTION THRESHOLDS
# ============================================================================

CONVERGENCE_LOSS_THRESHOLD = 1e-6
"""
Loss change threshold for convergence detection.

If loss change between epochs is below this threshold for multiple
consecutive epochs, training is considered converged.
"""

CONVERGENCE_PATIENCE = 5
"""
Number of epochs with minimal loss change to declare convergence.

Requires 5 consecutive epochs with loss change < threshold to avoid
false positives from noise.
"""

# ============================================================================
# DATASET-SPECIFIC CONSTANTS
# ============================================================================

# Dataset normalization constants (mean and standard deviation)
MNIST_MEAN = (0.1307,)
"""MNIST dataset mean for normalization (single channel grayscale)"""

MNIST_STD = (0.3081,)
"""MNIST dataset standard deviation for normalization (single channel grayscale)"""

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
"""CIFAR10 dataset mean for normalization (RGB channels)"""

CIFAR10_STD = (0.2023, 0.1994, 0.2010)
"""CIFAR10 dataset standard deviation for normalization (RGB channels)"""

# Dataset dimensions
MNIST_IMAGE_SIZE = 28
"""MNIST image dimensions (28x28 grayscale)"""

MNIST_NUM_CLASSES = 10
"""MNIST number of classes (digits 0-9)"""

CIFAR10_IMAGE_SIZE = 32
"""CIFAR-10 image dimensions (32x32 RGB)"""

CIFAR10_NUM_CLASSES = 10
"""CIFAR-10 number of classes"""

MEDICAL_DEFAULT_IMAGE_SIZE = 128
"""Default image size for medical segmentation (128x128)"""

NLP_MAX_SEQUENCE_LENGTH = 128
"""Default maximum sequence length for NLP tasks"""

# ============================================================================
# MODEL ARCHITECTURE CONSTANTS
# ============================================================================

RESNET18_PARAMS = 11_173_962
"""
Number of parameters in ResNet-18.

Used for model size comparisons and memory estimation.
"""

SIMPLE_CNN_PARAMS = 1_199_882
"""
Number of parameters in SimpleCNN (MNIST baseline).

Lightweight CNN for MNIST (2 conv layers + 2 FC layers).
"""

# ============================================================================
# STANDARDIZED NAMING CONSTANTS
# ============================================================================

class OptimizerNames:
    """Centralized optimizer name constants for consistent naming across codebase."""
    SGD = "SGD"
    SGD_MOMENTUM = "SGD_Momentum"
    SGD_NESTEROV = "SGD_Nesterov"
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSProp"
    SAM = "SAM"
    LOOKAHEAD = "Lookahead"
    AMSGRAD = "AMSGrad"
    ADABOUND = "AdaBound"
    RADAM = "RAdam"
    LAMB = "LAMB"

class DatasetNames:
    """Centralized dataset name constants for consistent naming across codebase."""
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    IMDB = "IMDB"
    MEDICAL = "Medical"

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_learning_rate(lr: float, optimizer_name: str) -> None:
    """
    Validate learning rate is within reasonable range for optimizer.
    
    Args:
        lr: Learning rate to validate
        optimizer_name: Name of optimizer
        
    Raises:
        ValueError: If learning rate is invalid
    """
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    
    # Optimizer-specific warnings
    if optimizer_name.lower() in ['sgd', 'sgd_momentum'] and lr < 0.001:
        import logging
        logging.warning(
            f"Learning rate {lr} is very low for SGD. "
            f"Consider using {SGD_DEFAULT_LR} or higher."
        )
    elif optimizer_name.lower() in ['adam', 'adamw'] and lr > 0.01:
        import logging
        logging.warning(
            f"Learning rate {lr} is very high for {optimizer_name}. "
            f"Consider using {ADAM_DEFAULT_LR} or lower."
        )


def validate_batch_size(batch_size: int, dataset: str) -> None:
    """
    Validate batch size is appropriate for dataset.
    
    Args:
        batch_size: Batch size to validate
        dataset: Dataset name
        
    Raises:
        ValueError: If batch size is invalid
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {batch_size}")
    
    # Check if batch size is power of 2 (GPU-friendly)
    if batch_size & (batch_size - 1) != 0:
        import logging
        logging.info(
            f"Batch size {batch_size} is not a power of 2. "
            f"GPU performance may be suboptimal. Consider using 32, 64, 128, etc."
        )
