# Changelog

## [Unreleased]

### Added - Infrastructure Enhancements (2026-02-02)
- **Checkpoint Management System** (`src/utils/checkpoint_utils.py`): Production-grade checkpoint handling with atomic writes, comprehensive metadata, and automatic cleanup
  - `save_checkpoint_atomic()`: Atomic saves with temp file + fsync + rename to prevent corruption
  - `create_checkpoint()`: Captures full metadata (model/optimizer state, config, git commit, RNG states for reproducibility)
  - `load_checkpoint_safe()`: Robust loading with validation, error handling, and RNG state restoration
  - `CheckpointManager`: Automatic cleanup with configurable retention (keep last N, keep best K, milestones)
  - Prevents checkpoint corruption even with process interruption
  - Full reproducibility across runs (Python/NumPy/PyTorch RNG states)
  - Git commit tracking for experiment provenance
  - Compatible with PyTorch 2.6+ (`weights_only` parameter handling)

- **Parallel Experiment Runner** (`src/utils/parallel_experiment_runner.py`): Multi-GPU support for Kaggle T4x2 with ~2x speedup
  - `ParallelExperimentRunner`: Queue-based parallel execution across multiple GPUs
  - `run_experiment_on_gpu()`: Worker function for isolated per-GPU execution
  - `detect_gpu_configuration()`: Automatic GPU detection and capability assessment
  - Graceful fallback to sequential execution if <2 GPUs available
  - Expected performance: 2x speedup on Kaggle T4x2 (2 GPUs)
  - Dynamic load balancing with multiprocessing queue
  - Per-experiment error handling (one failure doesn't stop others)

- **Resume Support Utilities** (`src/utils/resume_utils.py`): Intelligent experiment skip logic for long benchmark runs
  - `should_skip_experiment()`: Check if experiment already completed with validation
  - `validate_experiment_result()`: Validate result file integrity (epochs, columns, no NaN)
  - `count_completed_experiments()`: Summary statistics for progress tracking
  - Saves hours/days when re-running failed benchmarks
  - Safe fallback if result corrupted or incomplete

- **Optimizer Base Class Refactoring** (`src/core/optimizers.py`): Eliminated boilerplate with dispatch pattern
  - Added `_dispatch_step()` helper method to `Optimizer` base class
  - Generic dispatcher for tuple vs array parameter handling
  - Refactored SGD as example implementation
  - Reduces ~30 lines of boilerplate per optimizer
  - Pattern ready for remaining 11 optimizer classes (Adam, AdamW, RMSProp, SAM, etc.)

- **Comprehensive Testing** (`test_new_implementations.py`): Validation suite for all new features
  - Test 1: Optimizer dispatch pattern (tuple and array params)
  - Test 2: Checkpoint utilities (atomic save/load, CheckpointManager cleanup)
  - Test 3: Parallel runner GPU detection
  - Test 4: Resume utilities (validation and skip logic)
  - All tests passing ✅

- **Documentation**:
  - `IMPLEMENTATION_STATUS.md`: Detailed implementation status and integration guide
  - `IMPLEMENTATION_COMPLETE_SUMMARY.md`: Executive summary with metrics and validation
  - `docs/KAGGLE_T4X2_GUIDE.md`: Step-by-step guide for Kaggle notebook integration

### Fixed - Type Safety Phase 1 (2026-02-02)
- **CRITICAL TYPE SAFETY FIXES**: Implemented all 8 Phase 1 type safety fixes from TYPE_SAFETY_AUDIT_REPORT.md
  - **Fix 1**: Clarified optimizer `step()` return type documentation for all optimizers (SGD, Adam, AdamW, etc.) to ensure type preservation (tuple → tuple, ndarray → ndarray)
  - **Fix 2**: Replaced unsafe assertions with explicit None checks in Adam optimizer to prevent crashes with `python -O` (optimized mode). Changed `assert self.m is not None` to explicit `if self.m is None: raise TypeError(...)`
  - **Fix 3**: Added SAM optimizer API contract validation with clear, actionable error messages and usage examples when required parameters are missing
  - **Fix 4**: Verified all 11 PyTorch optimizer wrappers have consistent `Optional[float]` return types with proper Tensor-to-float conversion
  - **Fix 5**: Added explicit type annotations in training loops to separate `loss_tensor: torch.Tensor` from `loss_value: float` for clarity
  - **Fix 6**: Added `ExperimentTracker.active_run_id` property with validation to prevent unsafe Optional access and provide clear error messages
  - **Fix 7**: Improved `_safe_len()` exception handling by replacing bare `except:` with specific exception types `(TypeError, AttributeError)` and logging for unexpected errors
  - **Fix 8**: Added type guards (`hasattr(x, 'shape')`) before accessing `.shape` attribute in test files and model forward methods to prevent AttributeError
  - Created `verify_type_fixes.py` automated verification script (all 8 tests passing ✅)
  - See `TYPE_FIXES_PHASE1_COMPLETE.md` for complete implementation details
  - Updated `MASTER_FIX_TRACKER.md` to mark Phase 1 type fixes as DONE (8/216 issues resolved, 19% completion)

### Added
- Add `resume_behavior` option and robust resume checks: implement `src/core/resume_utils.py` for deterministic run signatures and results-aware resume decisions. Adds CLI flag `--resume-behavior` with choices `error_if_no_checkpoint`, `restart_if_no_checkpoint`, `skip_if_results_exist`. (PR: feature/bdnsca-ci)


### Fixed
- **DiskSpaceGuardian**: Corrected checkpoint cleanup logic to only delete checkpoints when the number of checkpoints exceeds `max_checkpoints`. Added tests to verify behavior. (src/core/training_enhancements.py, tests/test_disk_space_guardian.py)
- **Narrowed broad exception handling**: Replaced or annotated broad `except Exception:` usages in critical modules to avoid swallowing programming errors and to ensure IO/MLflow errors are handled explicitly. Files modified: `src/core/experiment_tracker.py` (narrowed MLflow catches and added intentional comments), `src/core/resume_utils.py` (narrowed CSV/serialization catches), `src/utils/csv_utils.py` (narrowed CSV parsing catches and improved corrupt CSV cleanup), `src/utils/file_safety.py` (narrowed I/O catches), and `run_all_kaggle.py` (added comments for intentional broad catches and narrowed some import catches). Added CI guard test `tests/test_broad_except_guard.py` to enforce intentional annotations, and unit tests covering MLflow failure modes and CSV/file I/O failures. (tests/test_experiment_tracker_mlflow_failures.py, tests/test_resume_utils.py, tests/test_csv_utils_and_file_safety.py)
- **Visualization fault isolation**: Add `plot_protect` context manager for plot-level failure isolation and update `src/visualization/cifar_viz.py` to use it, preventing a single plot failure from disabling further visualizations. Plot failures now log a WARNING and stack trace at DEBUG, and `plot_protect(strict=True)` will re-raise for debugging/CI. Added tests: `tests/test_visualization_fault_isolation.py`.
- **Retry utilities**: Narrowed default caught exceptions in `retry_with_backoff` and `retry_operation` to `NETWORK_EXCEPTIONS` (no longer catches all `Exception`, so signals like `KeyboardInterrupt` are not swallowed). Added tests for retry behavior. (src/core/retry.py, tests/test_retry.py)
- **ExperimentTracker**: Made `start_run()` defensive when `mlflow` returns a run-like object missing `info` or `run_id` attributes (returns `None` instead of raising). (src/core/experiment_tracker.py, tests/test_experiment_tracker_start_run.py)
- **Statistical analysis**: Use a local RNG in `cohens_d_ci_paired` to avoid mutating global NumPy RNG state and improve reproducibility. (src/analysis/statistical_analysis.py)
- **Checkpoint backups**: Improved lock acquisition to use atomic creation and remove stale locks older than 1 hour to prevent deadlocks and stale lock files. Added token-based lock ownership to ensure only the lock owner can release the lock; made lock timeouts and stale thresholds configurable. (src/core/checkpoint_manager.py)
- **Resume detection**: Robust boolean parsing in `results_exist()` to avoid treating string values like `'False'` as truthy, and improved numeric final metric checks to avoid treating `'nan'` strings as valid metrics. Added tests for these edge cases. (src/core/resume_utils.py, tests/test_resume_behavior_extended.py)
- **Runtime backups**: Unified atomic lock acquisition in `run_all_kaggle.py` to match `CheckpointManager` behavior and added stale-lock removal tests (tests/test_backup_lock_stale.py).
- **MLflow robustness**: Fail-safe ExperimentTracker creation (no longer crashes on MLflow DB/schema errors); logs remediation steps (e.g., `mlflow db upgrade <database_uri>`) and disables tracking when initialization fails. Added unit test to validate safe behavior. (run_all_kaggle.py, tests/test_mlflow_tracker_creation.py)
- **Notebooks**: Added `scripts/validate_notebooks.py` to scan notebooks for trailing backslashes and common syntax pitfalls that cause papermill execution errors, and `scripts/fix_and_validate_notebooks.py` to auto-fix literal '\\n' sequences and validate notebook structure before execution. (scripts/validate_notebooks.py, scripts/fix_and_validate_notebooks.py)
- **Warnings**: Suppress noisy SyntaxWarning messages from third-party Markdown/renderers during notebook conversion to reduce spurious logs. (run_all_kaggle.py)

### Testing
- Added unit tests for retry behavior, DiskSpaceGuardian cleanup, ExperimentTracker `start_run` edge case, resume boolean parsing, and backup-stale behavior. (tests/test_retry.py, tests/test_disk_space_guardian.py, tests/test_experiment_tracker_start_run.py, tests/test_resume_behavior_extended.py, tests/test_backup_lock_stale.py)
