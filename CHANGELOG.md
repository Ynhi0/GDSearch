# Changelog

## Unreleased

### Added
- Add `resume_behavior` option and robust resume checks: implement `src/core/resume_utils.py` for deterministic run signatures and results-aware resume decisions. Adds CLI flag `--resume-behavior` with choices `error_if_no_checkpoint`, `restart_if_no_checkpoint`, `skip_if_results_exist`. (PR: feature/bdnsca-ci)


### Fixed
- **DiskSpaceGuardian**: Corrected checkpoint cleanup logic to only delete checkpoints when the number of checkpoints exceeds `max_checkpoints`. Added tests to verify behavior. (src/core/training_enhancements.py, tests/test_disk_space_guardian.py)
- **Retry utilities**: Narrowed default caught exceptions in `retry_with_backoff` and `retry_operation` to `NETWORK_EXCEPTIONS` (no longer catches all `Exception`, so signals like `KeyboardInterrupt` are not swallowed). Added tests for retry behavior. (src/core/retry.py, tests/test_retry.py)
- **ExperimentTracker**: Made `start_run()` defensive when `mlflow` returns a run-like object missing `info` or `run_id` attributes (returns `None` instead of raising). (src/core/experiment_tracker.py, tests/test_experiment_tracker_start_run.py)
- **Statistical analysis**: Use a local RNG in `cohens_d_ci_paired` to avoid mutating global NumPy RNG state and improve reproducibility. (src/analysis/statistical_analysis.py)
- **Checkpoint backups**: Improved lock acquisition to use atomic creation and remove stale locks older than 1 hour to prevent deadlocks and stale lock files. (src/core/checkpoint_manager.py)

### Testing
- Added unit tests for retry behavior, DiskSpaceGuardian cleanup, and ExperimentTracker `start_run` edge case. (tests/test_retry.py, tests/test_disk_space_guardian.py, tests/test_experiment_tracker_start_run.py)
