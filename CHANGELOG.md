# Changelog

## Unreleased

### Added
- Add `resume_behavior` option and robust resume checks: implement `src/core/resume_utils.py` for deterministic run signatures and results-aware resume decisions. Adds CLI flag `--resume-behavior` with choices `error_if_no_checkpoint`, `restart_if_no_checkpoint`, `skip_if_results_exist`. (PR: feature/bdnsca-ci)


### Fixed
- **DiskSpaceGuardian**: Corrected checkpoint cleanup logic to only delete checkpoints when the number of checkpoints exceeds `max_checkpoints`. Added tests to verify behavior. (src/core/training_enhancements.py, tests/test_disk_space_guardian.py)
- **Narrowed broad exception handling**: Replaced or annotated broad `except Exception:` usages in critical modules to avoid swallowing programming errors and to ensure IO/MLflow errors are handled explicitly. Files modified: `src/core/experiment_tracker.py` (narrowed MLflow catches and added intentional comments), `src/core/resume_utils.py` (narrowed CSV/serialization catches), `src/utils/csv_utils.py` (narrowed CSV parsing catches and improved corrupt CSV cleanup), `src/utils/file_safety.py` (narrowed I/O catches), and `run_all_kaggle.py` (added comments for intentional broad catches and narrowed some import catches). Added CI guard test `tests/test_broad_except_guard.py` to enforce intentional annotations, and unit tests covering MLflow failure modes and CSV/file I/O failures. (tests/test_experiment_tracker_mlflow_failures.py, tests/test_resume_utils.py, tests/test_csv_utils_and_file_safety.py)
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
