"""
Centralized ExperimentTracker with safe MLflow integration.
This module ensures a single, well-typed implementation avoids optional-member
access warnings and provides consistent behavior across runners.
"""
from typing import Any, Dict, Optional
import logging
import os

try:
    # Import mlflow. In some environments (e.g., mismatched DB schema), importing mlflow
    # can raise runtime exceptions other than ImportError (like mlflow.exceptions.MlflowException).
    # Broad catch intentional: when mlflow is unavailable or raises runtime import errors
    # we prefer to disable experiment tracking instead of failing module import.
    import mlflow
    try:
        import mlflow.pytorch as mlflow_pytorch
    except (ImportError, ModuleNotFoundError):
        mlflow_pytorch = None
    HAS_MLFLOW = True
except Exception as e:  # broad catch intentional: mlflow import may raise MlflowException/runtime errors
    # Includes ImportError and runtime MlflowException raised during import
    mlflow = None  # type: ignore[assignment]
    mlflow_pytorch = None
    HAS_MLFLOW = False
    logging.warning("mlflow import failed (%s). Experiment tracking will be limited.", e)

import numpy as np
import torch


class ExperimentTracker:
    """Experiment tracking with MLflow integration.

    This class centralizes MLflow usage and performs explicit runtime guards
    to avoid optional-member access issues detected by static analyzers.
    """

    def __init__(self, experiment_name: str = "GDSearch_Benchmark", tracking_uri: Optional[str] = None, artifacts_dir: str = "artifacts"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.artifacts_dir = str(artifacts_dir)
        self.current_run = None
        self.run_stack = []  # type: list
        self.enabled = False

        # Early exit if MLflow not available -- avoids optional-member access
        if not (HAS_MLFLOW and mlflow is not None):
            logging.warning("mlflow not available. Experiment tracking disabled.")
            return

        # Try to configure MLflow (tracking URI + experiment). If this fails
        # (for example due to an out-of-date DB schema), we log a warning and
        # proceed with MLflow disabled so experiments don't crash.
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.enabled = True
        except (RuntimeError, OSError) as e:
            logging.warning("MLflow initialization failed (%s). Experiment tracking disabled.", e)
            # Do not raise; experiments should continue even if MLflow is misconfigured
            self.enabled = False
        except Exception as e:
            # If mlflow raised a domain-specific MlflowException, treat it as non-fatal and disable tracking;
            # otherwise re-raise programming errors so they surface during development.
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
            if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                # Check if this is a database schema error
                error_msg = str(e).lower()
                if 'schema' in error_msg or 'out-of-date' in error_msg or 'upgrade' in error_msg:
                    logging.warning("MLflow database schema is out of date. Attempting automatic upgrade...")
                    if self._attempt_db_upgrade(tracking_uri):
                        logging.info("MLflow database upgrade successful. Retrying initialization...")
                        try:
                            if tracking_uri:
                                mlflow.set_tracking_uri(tracking_uri)
                            mlflow.set_experiment(experiment_name)
                            self.enabled = True
                            logging.info("MLflow initialized successfully after database upgrade.")
                            return
                        except Exception as retry_error:
                            logging.warning("MLflow initialization still failed after upgrade: %s", retry_error)
                    else:
                        logging.warning("MLflow database upgrade failed or not attempted. Trying fresh database...")
                        if self._attempt_fresh_db(tracking_uri):
                            logging.info("Created fresh MLflow database. Retrying initialization...")
                            try:
                                if tracking_uri:
                                    mlflow.set_tracking_uri(tracking_uri)
                                mlflow.set_experiment(experiment_name)
                                self.enabled = True
                                logging.info("MLflow initialized successfully with fresh database.")
                                return
                            except Exception as retry_error:
                                logging.warning("MLflow initialization failed with fresh database: %s", retry_error)
                
                logging.warning("MLflow initialization failed (%s). Experiment tracking disabled.", e)
                logging.warning("Remediation: Run 'mlflow db upgrade <database_uri>' manually or use --no-mlflow flag.")
                self.enabled = False
            else:
                raise

    def _attempt_db_upgrade(self, tracking_uri: Optional[str]) -> bool:
        """Attempt to upgrade MLflow database schema automatically.
        
        Returns:
            True if upgrade succeeded, False otherwise
        """
        import subprocess
        import sys
        
        try:
            # Determine the database URI to upgrade
            db_uri = tracking_uri if tracking_uri else "mlruns"
            
            # In Kaggle environment, may not have write permissions
            if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
                logging.info("Running in Kaggle environment - skipping database upgrade (likely read-only filesystem)")
                return False
            
            logging.info(f"Attempting to upgrade MLflow database: {db_uri}")
            result = subprocess.run(
                [sys.executable, "-m", "mlflow", "db", "upgrade", db_uri],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logging.info("MLflow database upgrade completed successfully.")
                return True
            else:
                logging.warning(f"MLflow database upgrade failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.warning("MLflow database upgrade timed out after 30 seconds.")
            return False
        except Exception as e:
            logging.warning(f"Failed to run MLflow database upgrade: {e}")
            return False

    def _attempt_fresh_db(self, tracking_uri: Optional[str]) -> bool:
        """Attempt to create a fresh MLflow database by backing up and recreating.
        
        Returns:
            True if fresh database created successfully, False otherwise
        """
        import shutil
        from pathlib import Path
        from datetime import datetime
        
        try:
            # Determine the database path
            db_path = Path(tracking_uri if tracking_uri else "mlruns")
            
            # In Kaggle environment, may not have write permissions
            if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
                logging.info("Running in Kaggle environment - skipping database recreation (likely read-only filesystem)")
                return False
            
            if not db_path.exists():
                logging.info(f"Database path {db_path} does not exist, will be created fresh on next MLflow call.")
                return True
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = db_path.parent / f"{db_path.name}_backup_{timestamp}"
            
            logging.info(f"Backing up MLflow database from {db_path} to {backup_path}")
            shutil.move(str(db_path), str(backup_path))
            logging.info(f"Backup created successfully. Fresh database will be created at {db_path}")
            
            return True
            
        except Exception as e:
            logging.warning(f"Failed to create fresh MLflow database: {e}")
            return False

    def _resume_meta_path(self) -> str:
        """Path to the persisted resume metadata file (artifacts/resume_meta.json)."""
        return os.path.join(self.artifacts_dir, "resume_meta.json")

    def _write_resume_meta(self, run_id: Optional[str], checkpoint: Optional[str]):
        """Persist run_id and last checkpoint so future invocations can resume."""
        try:
            meta = {"run_id": run_id, "checkpoint": checkpoint}
            os.makedirs(self.artifacts_dir, exist_ok=True)
            with open(self._resume_meta_path(), "w", encoding="utf-8") as fh:
                import json
                json.dump(meta, fh)
        except Exception:
            # Best-effort persistence; do not raise to avoid breaking experiments
            logging.debug("Failed to write resume meta", exc_info=True)

    def _read_resume_meta(self) -> Optional[dict]:
        """Read persisted resume metadata if present, else None."""
        p = self._resume_meta_path()
        if not os.path.exists(p):
            return None
        try:
            import json
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logging.debug("Failed to read resume meta", exc_info=True)
            return None

    def register_checkpoint(self, path: str) -> None:
        """Register a checkpoint path with the tracker (updates persisted resume meta).

        Call this from checkpointing code after a successful save so future runs can resume.
        """
        try:
            run_id = None
            try:
                run_id = self.active_run_id
            except RuntimeError:
                run_id = None
            self._write_resume_meta(run_id, path)
            # Also log checkpoint as artifact when MLflow enabled
            if self.enabled and self.current_run is not None:
                try:
                    mlflow.log_artifact(path)
                except Exception:
                    logging.debug("Failed to log checkpoint artifact to MLflow", exc_info=True)
        except Exception:
            logging.debug("register_checkpoint failed", exc_info=True)

    @property
    def active_run_id(self) -> str:
        """Get active run ID, raises if no active run.
        
        TYPE SAFETY FIX: Property with validation to prevent Optional[str] access issues.
        Use this instead of self.current_run.info.run_id to ensure run is active.
        
        Returns:
            Active MLflow run ID
            
        Raises:
            RuntimeError: If no active MLflow run
        """
        if self.current_run is None:
            raise RuntimeError(
                "No active MLflow run. Call start_run() before logging metrics/parameters.\n"
                "Example:\n"
                "  tracker = ExperimentTracker()\n"
                "  tracker.start_run(run_name='my_experiment')\n"
                "  tracker.log_metric('loss', 0.5)\n"
                "  tracker.end_run()"
            )
        info = getattr(self.current_run, "info", None)
        if info is None:
            raise RuntimeError("Current run has no info attribute")
        run_id = getattr(info, "run_id", None)
        if run_id is None:
            raise RuntimeError("Current run info has no run_id attribute")
        return run_id

    def start_run(self, run_name: Optional[str] = None, resume: bool = False) -> Optional[str]:
        """Start a new MLflow run (supports nested runs). If `resume=True`, attempt to attach
        to a previously persisted run using `artifacts/resume_meta.json`.
        """
        if not self.enabled:
            return None

        # If resume requested, prefer persisted run_id when available
        if resume:
            try:
                meta = self._read_resume_meta()
                if meta and meta.get('run_id'):
                    # Attach to existing run by run_id
                    try:
                        self.current_run = mlflow.start_run(run_id=meta.get('run_id'))
                    except Exception:
                        logging.debug("Failed to attach to persisted run_id; will start a new run", exc_info=True)
            except Exception:
                logging.debug("Failed to read resume metadata; starting fresh run", exc_info=True)

        if self.current_run is not None:
            # Start a nested/child run
            self.run_stack.append(self.current_run)
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
            try:
                self.current_run = mlflow.start_run(run_name=run_name, nested=True)
            except RuntimeError:
                # Restore stack on failure to prevent corruption
                self.run_stack.pop()
                raise
            except Exception as e:
                # Broad catch intentional: ensure run_stack is restored for any raised mlflow-specific error
                self.run_stack.pop()
                if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                    raise
                # Unexpected exception types: re-raise to surface programming errors
                raise
        else:
            # Start a new top-level run if not already attached via resume
            try:
                self.current_run = mlflow.start_run(run_name=run_name)
            except Exception as e:
                # Surface Mlflow-specific exceptions, otherwise re-raise
                mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
                mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
                if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                    raise
                raise

        if self.current_run is None:
            return None

        # Persist run_id so subsequent invocations may resume when requested
        try:
            info = getattr(self.current_run, "info", None)
            run_id = getattr(info, "run_id", None)
            # register persisted run_id, checkpoint unknown until register_checkpoint is called
            self._write_resume_meta(run_id, None)
        except Exception:
            logging.debug("Failed to persist resume meta after start_run", exc_info=True)

        info = getattr(self.current_run, "info", None)
        return getattr(info, "run_id", None)

    def end_run(self):
        """End the current MLflow run."""
        if not (self.enabled and self.current_run):
            return

        try:
            mlflow.end_run()
        except (RuntimeError, OSError) as e:
            logging.exception("Failed to end mlflow run: %s", e)
        except Exception as e:
            # Broad catch intentional: mlflow might raise domain-specific exceptions; log and continue
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
            if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                logging.exception("Failed to end mlflow run: %s", e)
            else:
                raise
        if self.run_stack:
            # Restore parent run
            self.current_run = self.run_stack.pop()
        else:
            self.current_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters, converting non-serializable types to strings."""
        if not (self.enabled and self.current_run):
            return

        for k, v in params.items():
            # Handle numpy/torch types explicitly
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None

            if isinstance(v, (np.ndarray,)):
                try:
                    elem_count = int(v.size)
                except (TypeError, AttributeError, ValueError):
                    elem_count = None
                if elem_count is not None and elem_count <= 100:
                    try:
                        v = v.tolist()
                    except (AttributeError, TypeError, ValueError):
                        v = str(v)
                else:
                    v = f"<{type(v).__name__} shape={getattr(v, 'shape', None)}>"
            elif isinstance(v, torch.Tensor):
                try:
                    elem_count = int(v.numel())
                except (TypeError, AttributeError, ValueError):
                    elem_count = None
                if elem_count is not None and elem_count <= 100:
                    try:
                        v = v.tolist()
                    except (AttributeError, TypeError, ValueError):
                        v = str(v)
                else:
                    v = f"<{type(v).__name__} shape={getattr(v, 'shape', None)}>"
            elif isinstance(v, (list, tuple, dict)):
                v = str(v)
            elif v is None:
                v = "None"
            elif not isinstance(v, (str, int, float, bool)):
                v = str(v)

            try:
                mlflow.log_param(k, v)
            except (TypeError, ValueError, RuntimeError, OSError) as e:
                logging.exception("Failed to log param %s=%s: %s", k, v, e)
            except Exception as e:
                # Broad catch intentional: mlflow may raise a domain-specific MlflowException; log and continue
                if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                    logging.exception("Failed to log param %s=%s: %s", k, v, e)
                else:
                    raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not (self.enabled and self.current_run):
            return

        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, v, step=step)
            except (TypeError, ValueError, RuntimeError, OSError) as e:
                logging.exception("Failed to log metric %s=%s: %s", k, v, e)
            except Exception as e:
                # Broad catch intentional: mlflow may raise a domain-specific MlflowException; log and continue
                mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
                mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
                if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                    logging.exception("Failed to log metric %s=%s: %s", k, v, e)
                else:
                    raise

    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model via mlflow.pytorch when available."""
        if not (self.enabled and self.current_run):
            return

        try:
            if mlflow_pytorch is not None:
                mlflow_pytorch.log_model(model, model_name)
            else:
                logging.warning("mlflow.pytorch not available; skipping model logging")
        except (RuntimeError, OSError, TypeError, ValueError) as e:
            logging.warning("Failed to log model to MLflow: %s", e)
        except Exception as e:
            # Broad catch intentional: mlflow.pytorch may raise domain specific exceptions
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
            if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                logging.warning("Failed to log model to MLflow: %s", e)
            else:
                raise

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        if not (self.enabled and self.current_run):
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except (OSError, RuntimeError, ValueError) as e:
            logging.exception("Failed to log artifact %s -> %s: %s", local_path, artifact_path, e)
        except Exception as e:
            # Broad catch intentional: mlflow may raise domain-specific exceptions; log and continue
            mlflow_exc_mod = getattr(mlflow, 'exceptions', None)
            mlflow_exc_cls = getattr(mlflow_exc_mod, 'MlflowException', None) if mlflow_exc_mod is not None else None
            if mlflow_exc_cls is not None and isinstance(e, mlflow_exc_cls):
                logging.exception("Failed to log artifact %s -> %s: %s", local_path, artifact_path, e)
            else:
                raise
