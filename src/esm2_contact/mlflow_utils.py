"""
MLflow Utilities for ESM2 Contact Prediction

This module provides MLflow integration utilities following 2025 best practices
for experiment tracking, hyperparameter optimization, and model serving.

Key Features:
- Context manager integration for proper resource management
- PyTorch autologging with exclusivity
- Structured parameter and metric logging
- Artifact management with organization
- Git integration for reproducibility
- Model versioning and registry support

Usage:
    from esm2_contact.mlflow_utils import MLflowTracker, setup_mlflow

    # Setup MLflow tracking
    setup_mlflow(tracking_uri="file:///tmp/mlruns")

    # Track experiment
    with MLflowTracker("experiment_name") as tracker:
        tracker.log_params({"learning_rate": 0.001})
        tracker.log_metrics({"val_auc": 0.85})
        tracker.log_model(model, "model")
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import mlflow
import mlflow.pytorch
import mlflow.utils
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment
import torch

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    warnings.warn("Git not available. Install gitpython for better reproducibility.")


class MLflowTracker:
    """
    MLflow tracking wrapper following 2025 best practices.

    Provides context manager integration, structured logging, and artifact management
    for ESM2 contact prediction experiments.

    Args:
        experiment_name (str): Name of the MLflow experiment
        run_name (Optional[str]): Specific run name, auto-generated if None
        tags (Optional[Dict[str, str]]): Tags for experiment organization
        tracking_uri (Optional[str]): MLflow tracking URI
        log_git_info (bool): Whether to log git information

    Example:
        with MLflowTracker("protein_contact_prediction") as tracker:
            tracker.log_params({"learning_rate": 0.001, "batch_size": 4})
            # ... training logic ...
            tracker.log_metrics({"val_auc": 0.85, "test_auc": 0.83})
            tracker.log_model(model, "contact_model")
    """

    def __init__(self,
                 experiment_name: str,
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None,
                 tracking_uri: Optional[str] = None,
                 log_git_info: bool = True):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tags = tags or {}
        self.tracking_uri = tracking_uri
        self.log_git_info = log_git_info
        self.run = None
        self.client = None

        # Default tags for ESM2 contact prediction
        default_tags = {
            "project": "esm2_contact_prediction",
            "model_type": "binary_cnn",
            "framework": "pytorch",
            "year": "2025"
        }
        default_tags.update(self.tags)
        self.tags = default_tags

    def __enter__(self):
        """Context manager entry - start MLflow run."""
        try:
            # Set tracking URI if provided
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
                self.client = MlflowClient(tracking_uri=self.tracking_uri)

            # Start run with context manager (2025 best practice)
            # Get or create experiment by name, then use experiment_id
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                # Create experiment if it doesn't exist
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id

            self.run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=self.run_name,
                tags=self.tags
            )

            # Log git information if available
            if self.log_git_info:
                self._log_git_info()

            # Log system information
            self._log_system_info()

            return self

        except Exception as e:
            # Enhanced error reporting for debugging
            error_msg = f"Failed to start MLflow run: {e}"
            print(f"❌ {error_msg}")
            print(f"   Experiment name: {self.experiment_name}")
            print(f"   Run name: {self.run_name}")
            print(f"   Tracking URI: {self.tracking_uri}")
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"   Root cause: {e.__cause__}")
            warnings.warn(error_msg)
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end MLflow run."""
        if self.run:
            try:
                # Log run completion status
                if exc_type is None:
                    mlflow.set_tag("status", "completed")
                else:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(exc_val))

                mlflow.end_run()
                self.run = None

            except Exception as e:
                warnings.warn(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow with structured organization.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to log
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log parameters.")
            return

        try:
            # Structure parameters by category
            structured_params = {}

            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    structured_params[key] = value
                else:
                    # Convert complex objects to JSON strings
                    structured_params[key] = json.dumps(value, default=str)

            mlflow.log_params(structured_params)

        except Exception as e:
            warnings.warn(f"Failed to log parameters: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log
            step (Optional[int]): Step number for metrics
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log metrics.")
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            warnings.warn(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """
        Log artifact to MLflow.

        Args:
            local_path (Union[str, Path]): Path to local file or directory
            artifact_path (Optional[str]): Destination path in MLflow
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log artifact.")
            return

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            warnings.warn(f"Failed to log artifact {local_path}: {e}")

    def log_model(self, model: torch.nn.Module, artifact_path: str,
                   input_example: Optional[torch.Tensor] = None,
                   registered_model_name: Optional[str] = None):
        """
        Log PyTorch model to MLflow.

        Args:
            model (torch.nn.Module): PyTorch model to log
            artifact_path (str): Path for model artifact
            input_example (Optional[torch.Tensor]): Example input for model signature
            registered_model_name (Optional[str]): Name for model registry
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log model.")
            return

        try:
            # Log model info
            model_info = self._get_model_info(model)
            self.log_params({"model_info": model_info})

            # Log model with signature if input example provided
            if input_example is not None:
                from mlflow.models.signature import infer_signature
                signature = infer_signature(input_example, model(input_example))

                mlflow.pytorch.log_model(
                    model,
                    artifact_path=artifact_path,
                    signature=signature,
                    registered_model_name=registered_model_name
                )
            else:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )

        except Exception as e:
            warnings.warn(f"Failed to log model: {e}")

    def log_training_history(self, history: Dict[str, Any], artifact_path: str = "training_history"):
        """
        Log training history as JSON artifact.

        Args:
            history (Dict[str, Any]): Training history dictionary
            artifact_path (str): Path for artifact
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log training history.")
            return

        try:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, value in history.items():
                if hasattr(value, 'tolist'):
                    json_history[key] = value.tolist()
                else:
                    json_history[key] = value

            # Save as temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(json_history, f, indent=2)
                temp_path = f.name

            # Log as artifact
            self.log_artifact(temp_path, artifact_path)

            # Clean up temporary file
            os.unlink(temp_path)

        except Exception as e:
            warnings.warn(f"Failed to log training history: {e}")

    def log_dataset_info(self, dataset_path: str, dataset_info: Dict[str, Any]):
        """
        Log dataset information.

        Args:
            dataset_path (str): Path to dataset
            dataset_info (Dict[str, Any]): Dataset information
        """
        if not self.run:
            warnings.warn("No active MLflow run. Cannot log dataset info.")
            return

        try:
            # Log dataset parameters
            dataset_params = {
                "dataset_path": dataset_path,
                **dataset_info
            }
            self.log_params(dataset_params)

            # Log dataset file if it exists
            if Path(dataset_path).exists():
                self.log_artifact(dataset_path, "dataset")

        except Exception as e:
            warnings.warn(f"Failed to log dataset info: {e}")

    def _log_git_info(self):
        """Log git repository information."""
        if not GIT_AVAILABLE:
            return

        try:
            repo = git.Repo(search_parent_directories=True)

            git_info = {
                "git_commit": repo.head.commit.hexsha,
                "git_branch": repo.active_branch.name,
                "git_dirty": repo.is_dirty(),
                "git_remote_url": None
            }

            # Get remote URL
            try:
                origin = repo.remotes.origin
                git_info["git_remote_url"] = origin.url
            except:
                pass

            self.log_params(git_info)

        except Exception as e:
            warnings.warn(f"Failed to log git info: {e}")

    def _log_system_info(self):
        """Log system information."""
        try:
            import platform
            import torch

            system_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "mlflow_version": mlflow.__version__
            }

            if torch.cuda.is_available():
                system_info["gpu_name"] = torch.cuda.get_device_name(0)
                system_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

            self.log_params(system_info)

        except Exception as e:
            warnings.warn(f"Failed to log system info: {e}")

    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Extract model information."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
                "model_type": type(model).__name__
            }
        except Exception as e:
            warnings.warn(f"Failed to get model info: {e}")
            return {}


def setup_mlflow(tracking_uri: Optional[str] = None,
                 experiment_name: str = "esm2_contact_prediction",
                 registry_uri: Optional[str] = None):
    """
    Setup MLflow tracking configuration.

    Args:
        tracking_uri (Optional[str]): MLflow tracking URI
        experiment_name (str): Default experiment name
        registry_uri (Optional[str]): MLflow model registry URI

    Returns:
        str: Configured tracking URI
    """
    try:
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local file-based tracking
            default_uri = "file:./mlruns"
            mlflow.set_tracking_uri(default_uri)
            tracking_uri = default_uri

        # Set registry URI if provided
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        # Create experiment if it doesn't exist
        client = MlflowClient(tracking_uri=tracking_uri)

        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                client.create_experiment(
                    name=experiment_name,
                    tags={"project": "esm2_contact_prediction", "year": "2025"}
                )
                print(f"✅ Created MLflow experiment: {experiment_name}")
            else:
                print(f"✅ Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            warnings.warn(f"Failed to create experiment: {e}")

        print(f"🔧 MLflow tracking configured: {tracking_uri}")
        return tracking_uri

    except Exception as e:
        warnings.warn(f"Failed to setup MLflow: {e}")
        return None


def enable_pytorch_autolog(log_models: bool = True,
                           exclusive: bool = True,
                           disable_for_unsupported_versions: bool = False):
    """
    Enable PyTorch autologging following 2025 best practices.

    Args:
        log_models (bool): Whether to log models automatically
        exclusive (bool): Whether to use exclusive autologging
        disable_for_unsupported_versions (bool): Disable for unsupported versions
    """
    try:
        mlflow.pytorch.autolog(
            log_models=log_models,
            exclusive=exclusive,
            disable_for_unsupported_versions=disable_for_unsupported_versions
        )
        print("✅ PyTorch autologging enabled")

    except Exception as e:
        warnings.warn(f"Failed to enable PyTorch autologging: {e}")


def get_experiment_runs(experiment_name: str,
                        tracking_uri: Optional[str] = None) -> list:
    """
    Get all runs for a specific experiment.

    Args:
        experiment_name (str): Name of the experiment
        tracking_uri (Optional[str]): MLflow tracking URI

    Returns:
        list: List of run information
    """
    try:
        if tracking_uri:
            client = MlflowClient(tracking_uri=tracking_uri)
        else:
            client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        runs = client.search_runs(experiment.experiment_id)
        return runs

    except Exception as e:
        warnings.warn(f"Failed to get experiment runs: {e}")
        return []


def get_best_run(experiment_name: str,
                 metric_name: str = "val_auc",
                 tracking_uri: Optional[str] = None) -> Optional[Any]:
    """
    Get the best run for an experiment based on a metric.

    Args:
        experiment_name (str): Name of the experiment
        metric_name (str): Name of the metric to optimize
        tracking_uri (Optional[str]): MLflow tracking URI

    Returns:
        Optional[Any]: Best run information or None
    """
    try:
        runs = get_experiment_runs(experiment_name, tracking_uri)

        if not runs:
            return None

        # Find run with best metric value
        best_run = None
        best_value = float('-inf')

        for run in runs:
            metrics = run.data.metrics
            if metric_name in metrics and metrics[metric_name] > best_value:
                best_value = metrics[metric_name]
                best_run = run

        return best_run

    except Exception as e:
        warnings.warn(f"Failed to get best run: {e}")
        return None