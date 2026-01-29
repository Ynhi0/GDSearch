"""
Centralized ExperimentTracker with safe MLflow integration.
This module ensures a single, well-typed implementation avoids optional-member
access warnings and provides consistent behavior across runners.
"""
from typing import Any, Dict, Optional
import logging

try:
    import mlflow
    try:
        import mlflow.pytorch as mlflow_pytorch
    except Exception:
        mlflow_pytorch = None
    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    mlflow_pytorch = None
    HAS_MLFLOW = False
    logging.warning("mlflow not available. Experiment tracking will be limited.")

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
        except Exception as e:
            logging.warning("MLflow initialization failed (%s). Experiment tracking disabled.", e)
            # Do not raise; experiments should continue even if MLflow is misconfigured
            self.enabled = False

    def start_run(self, run_name: Optional[str] = None) -> Optional[str]:
        """Start a new MLflow run (supports nested runs)."""
        if not self.enabled:
            return None

        if self.current_run is not None:
            # Start a nested/child run
            self.run_stack.append(self.current_run)
            try:
                self.current_run = mlflow.start_run(run_name=run_name, nested=True)
            except (Exception, RuntimeError):
                # Restore stack on failure to prevent corruption
                self.run_stack.pop()
                raise
        else:
            # Start a new top-level run
            self.current_run = mlflow.start_run(run_name=run_name)
        return getattr(self.current_run, "info", None).run_id if self.current_run is not None else None

    def end_run(self):
        """End the current MLflow run."""
        if not (self.enabled and self.current_run):
            return

        try:
            mlflow.end_run()
        except Exception:
            logging.exception("Failed to end mlflow run")
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
            if isinstance(v, (np.ndarray,)):
                try:
                    elem_count = int(v.size)
                except Exception:
                    elem_count = None
                if elem_count is not None and elem_count <= 100:
                    try:
                        v = v.tolist()
                    except Exception:
                        v = str(v)
                else:
                    v = f"<{type(v).__name__} shape={getattr(v, 'shape', None)}>"
            elif isinstance(v, torch.Tensor):
                try:
                    elem_count = int(v.numel())
                except Exception:
                    elem_count = None
                if elem_count is not None and elem_count <= 100:
                    try:
                        v = v.tolist()
                    except Exception:
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
            except Exception:
                logging.exception("Failed to log param %s=%s", k, v)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not (self.enabled and self.current_run):
            return

        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, v, step=step)
            except Exception:
                logging.exception("Failed to log metric %s=%s", k, v)

    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Log model via mlflow.pytorch when available."""
        if not (self.enabled and self.current_run):
            return

        try:
            if mlflow_pytorch is not None:
                mlflow_pytorch.log_model(model, model_name)
            else:
                logging.warning("mlflow.pytorch not available; skipping model logging")
        except Exception as e:
            logging.warning("Failed to log model to MLflow: %s", e)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file."""
        if not (self.enabled and self.current_run):
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception:
            logging.exception("Failed to log artifact %s -> %s", local_path, artifact_path)
