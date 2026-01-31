"""
Centralized ExperimentTracker with safe MLflow integration.
This module ensures a single, well-typed implementation avoids optional-member
access warnings and provides consistent behavior across runners.
"""
from typing import Any, Dict, Optional
import logging

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

    def __init__(self, experiment_name: str = "GDSearch_Benchmark", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
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
                logging.warning("MLflow initialization failed (%s). Experiment tracking disabled.", e)
                self.enabled = False
            else:
                raise

    def start_run(self, run_name: Optional[str] = None) -> Optional[str]:
        """Start a new MLflow run (supports nested runs)."""
        if not self.enabled:
            return None

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
            # Start a new top-level run
            self.current_run = mlflow.start_run(run_name=run_name)
        if self.current_run is None:
            return None
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
