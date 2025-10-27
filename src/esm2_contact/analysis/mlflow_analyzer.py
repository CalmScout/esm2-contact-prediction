"""
MLflow Experiment Analyzer for ESM2 Contact Prediction

This module provides utilities for extracting and analyzing MLflow experiment data,
including metrics extraction, run comparison, and performance analysis.

Key Features:
- Load experiment data from MLflow tracking server
- Extract training metrics and parameters
- Compare multiple experiments and runs
- Generate performance summaries
- Access model artifacts and configurations

Usage:
    from esm2_contact.analysis import MLflowAnalyzer

    analyzer = MLflowAnalyzer()
    run_data = analyzer.load_experiment("protein_contact_prediction")
    best_run = analyzer.get_best_run(run_data, metric="val_auc")
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting functionality will be limited.")


class MLflowAnalyzer:
    """
    Comprehensive MLflow experiment analyzer.

    Provides utilities for loading, analyzing, and comparing MLflow experiments
    with focus on ESM2 contact prediction models.

    Attributes:
        client (MlflowClient): MLflow client for data access
        tracking_uri (str): MLflow tracking URI

    Example:
        analyzer = MLflowAnalyzer()
        experiment_data = analyzer.load_experiment("contact_prediction")
        best_run = analyzer.get_best_run(experiment_data)
        comparison = analyzer.compare_runs([run1, run2])
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow analyzer.

        Args:
            tracking_uri (Optional[str]): MLflow tracking URI (default: file:///tmp/mlruns)
        """
        self.tracking_uri = tracking_uri or "file:///tmp/mlruns"
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments.

        Returns:
            List[Dict[str, Any]]: List of experiment information
        """
        try:
            experiments = self.client.search_experiments()
            exp_list = []

            for exp in experiments:
                exp_info = {
                    'experiment_id': exp.experiment_id,
                    'name': exp.name,
                    'creation_time': exp.creation_time,
                    'last_update': exp.last_update,
                    'tags': exp.tags if exp.tags else {},
                    'lifecycle_stage': exp.lifecycle_stage,
                }
                exp_list.append(exp_info)

            return exp_list

        except Exception as e:
            raise RuntimeError(f"Failed to list experiments: {e}")

    def load_experiment(self, experiment_name: str, include_artifacts: bool = True) -> Dict[str, Any]:
        """
        Load complete experiment data including runs, metrics, parameters, and artifacts.

        Args:
            experiment_name (str): Name of the experiment
            include_artifacts (bool): Whether to load artifact information

        Returns:
            Dict[str, Any]: Complete experiment data
        """
        try:
            print(f"🔍 Loading experiment: {experiment_name}")

            # Get experiment
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment not found: {experiment_name}")

            # Get all runs
            runs = self.client.search_runs(experiment.experiment_id)

            experiment_data = {
                'experiment_info': {
                    'experiment_id': experiment.experiment_id,
                    'name': experiment.name,
                    'creation_time': experiment.creation_time,
                    'tags': experiment.tags if experiment.tags else {}
                },
                'runs': {},
                'metadata': {
                    'total_runs': len(runs),
                    'loaded_at': datetime.now().isoformat(),
                    'tracking_uri': self.tracking_uri
                }
            }

            print(f"   Found {len(runs)} runs")

            # Process each run
            for run in runs:
                run_data = self._process_run(run, include_artifacts)
                experiment_data['runs'][run.info.run_id] = run_data

            print(f"✅ Loaded experiment: {experiment_name}")
            return experiment_data

        except Exception as e:
            raise RuntimeError(f"Failed to load experiment {experiment_name}: {e}")

    def _process_run(self, run, include_artifacts: bool = True) -> Dict[str, Any]:
        """Process a single run and extract all relevant data."""
        try:
            run_id = run.info.run_id
            run_data = {
                'run_info': {
                    'run_id': run_id,
                    'experiment_id': run.info.experiment_id,
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'lifecycle_stage': run.info.lifecycle_stage,
                    'run_name': run.info.run_name,
                    'user_id': run.info.user_id,
                    'tags': run.data.tags if run.data.tags else {}
                },
                'params': run.data.params if run.data.params else {},
                'metrics': run.data.metrics if run.data.metrics else {},
                'artifacts': {}
            }

            # Load training history artifact if available
            if include_artifacts:
                try:
                    artifacts = self.client.list_artifacts(run_id)
                    for artifact in artifacts:
                        if artifact.path.endswith('training_history.npy'):
                            run_data['artifacts']['training_history'] = self._load_training_history(
                                run_id, artifact.path
                            )
                        elif artifact.path.endswith('config.json'):
                            run_data['artifacts']['config'] = self._load_config(
                                run_id, artifact.path
                            )
                        elif artifact.path.endswith('model.pth'):
                            run_data['artifacts']['model_path'] = artifact.path

                except Exception as e:
                    warnings.warn(f"Failed to load artifacts for run {run_id}: {e}")

            return run_data

        except Exception as e:
            warnings.warn(f"Failed to process run {run.info.run_id}: {e}")
            return {}

    def _load_training_history(self, run_id: str, artifact_path: str) -> Dict[str, np.ndarray]:
        """Load training history from artifact."""
        try:
            local_path = self.client.download_artifacts(run_id, artifact_path)
            history_data = np.load(local_path, allow_pickle=True)

            # Convert to dictionary if needed
            if isinstance(history_data, np.ndarray) and history_data.dtype == object:
                # This is a structured array
                history = {}
                for field in history_data.dtype.names:
                    history[field] = history_data[field]
                return history
            elif isinstance(history_data, dict):
                return history_data
            else:
                # Assume it's already in the right format
                return history_data

        except Exception as e:
            warnings.warn(f"Failed to load training history: {e}")
            return {}

    def _load_config(self, run_id: str, artifact_path: str) -> Dict[str, Any]:
        """Load configuration from artifact."""
        try:
            local_path = self.client.download_artifacts(run_id, artifact_path)
            with open(local_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            warnings.warn(f"Failed to load config: {e}")
            return {}

    def get_run(self, experiment_name: str, run_id: str) -> Dict[str, Any]:
        """Get specific run data."""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment not found: {experiment_name}")

            run = self.client.get_run(run_id)
            return self._process_run(run)

        except Exception as e:
            raise RuntimeError(f"Failed to get run {run_id}: {e}")

    def get_best_run(self, experiment_data: Dict[str, Any],
                    metric: str = "val_auc", direction: str = "max") -> Dict[str, Any]:
        """
        Get the best performing run for an experiment.

        Args:
            experiment_data (Dict[str, Any]): Experiment data from load_experiment()
            metric (str): Metric to optimize (default: "val_auc")
            direction (str): Optimization direction ("max" or "min", default: "max")

        Returns:
            Dict[str, Any]: Best run data
        """
        try:
            best_run = None
            best_value = float('-inf') if direction == "max" else float('inf')

            for run_id, run_data in experiment_data['runs'].items():
                if metric in run_data['metrics']:
                    value = run_data['metrics'][metric]
                    if (direction == "max" and value > best_value) or \
                       (direction == "min" and value < best_value):
                        best_value = value
                        best_run = run_data

            if best_run:
                print(f"🏆 Best run found: {best_run['run_info']['run_id']}")
                print(f"   {metric}: {best_value:.4f}")
                print(f"   Status: {best_run['run_info']['status']}")
                return best_run
            else:
                print(f"⚠️ No runs found with metric '{metric}'")
                return {}

        except Exception as e:
            raise RuntimeError(f"Failed to find best run: {e}")

    def get_latest_run(self, experiment_name: str) -> Dict[str, Any]:
        """Get the most recent run for an experiment."""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment not found: {experiment_name}")

            runs = self.client.search_runs(
                experiment.experiment_id,
                order_by=["start_time DESC"],
                max_results=1
            )

            if runs:
                latest_run = runs[0]
                return self._process_run(latest_run)
            else:
                print(f"⚠️ No runs found for experiment: {experiment_name}")
                return {}

        except Exception as e:
            raise RuntimeError(f"Failed to get latest run: {e}")

    def compare_runs(self, run_data_list: List[Dict[str, Any]],
                    metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple runs and create a comparison table.

        Args:
            run_data_list (List[Dict[str, Any]]): List of run data
            metrics (Optional[List[str]]): Metrics to compare

        Returns:
            pd.DataFrame: Comparison table
        """
        try:
            if not run_data_list:
                return pd.DataFrame()

            # Default metrics to compare
            if metrics is None:
                metrics = ['val_auc', 'val_precision_at_l', 'val_precision_at_l5',
                          'train_loss', 'val_loss', 'total_epochs']

            comparison_data = []

            for run_data in run_data_list:
                run_info = run_data.get('run_info', {})
                row = {
                    'run_id': run_info.get('run_id', 'unknown'),
                    'run_name': run_info.get('run_name', 'unknown'),
                    'status': run_info.get('status', 'unknown'),
                    'start_time': run_info.get('start_time'),
                    'end_time': run_info.get('end_time')
                }

                # Add metrics
                for metric in metrics:
                    if metric in run_data.get('metrics', {}):
                        row[metric] = run_data['metrics'][metric]
                    else:
                        row[metric] = None

                # Add parameters
                for param, value in run_data.get('params', {}).items():
                    if param not in row:  # Avoid overwriting existing columns
                        row[param] = value

                comparison_data.append(row)

            df = pd.DataFrame(comparison_data)

            # Sort by primary metric if available
            if 'val_auc' in df.columns:
                df = df.sort_values('val_auc', ascending=False)
            elif metrics and metrics[0] in df.columns:
                df = df.sort_values(metrics[0],
                                ascending=False if metrics[0].startswith('val') else True)

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to compare runs: {e}")

    def analyze_experiment_overview(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overview statistics for an experiment.

        Args:
            experiment_data (Dict[str, Any]): Experiment data

        Returns:
            Dict[str, Any]: Overview statistics
        """
        try:
            runs = experiment_data.get('runs', {})

            # Basic statistics
            total_runs = len(runs)
            completed_runs = len([r for r in runs.values()
                                if r.get('run_info', {}).get('status') == 'FINISHED'])
            failed_runs = len([r for r in runs.values()
                              if r.get('run_info', {}).get('status') == 'FAILED'])

            # Performance metrics
            val_aucs = []
            precisions_l = []
            precisions_l5 = []

            for run_data in runs.values():
                metrics = run_data.get('metrics', {})
                if 'val_auc' in metrics:
                    val_aucs.append(metrics['val_auc'])
                if 'val_precision_at_l' in metrics:
                    precisions_l.append(metrics['val_precision_at_l'])
                if 'val_precision_at_l5' in metrics:
                    precisions_l5.append(metrics['val_precision_at_l5'])

            overview = {
                'total_runs': total_runs,
                'completed_runs': completed_runs,
                'failed_runs': failed_runs,
                'success_rate': completed_runs / total_runs if total_runs > 0 else 0,
                'performance': {
                    'val_auc': {
                        'mean': np.mean(val_aucs) if val_aucs else None,
                        'std': np.std(val_aucs) if val_aucs else None,
                        'min': np.min(val_aucs) if val_aucs else None,
                        'max': np.max(val_aucs) if val_aucs else None,
                        'count': len(val_aucs)
                    },
                    'val_precision_at_l': {
                        'mean': np.mean(precisions_l) if precisions_l else None,
                        'std': np.std(precisions_l) if precisions_l else None,
                        'min': np.min(precisions_l) if precisions_l else None,
                        'max': np.max(precisions_l) if precisions_l else None,
                        'count': len(precisions_l)
                    },
                    'val_precision_at_l5': {
                        'mean': np.mean(precisions_l5) if precisions_l5 else None,
                        'std': np.std(precisions_l5) if precisions_l5 else None,
                        'min': np.min(precisions_l5) if precisions_l5 else None,
                        'max': np.max(precisions_l5) if precisions_l5 else None,
                        'count': len(precisions_l5)
                    }
                }
            }

            return overview

        except Exception as e:
            raise RuntimeError(f"Failed to analyze experiment overview: {e}")

    def export_experiment_data(self, experiment_data: Dict[str, Any],
                              output_path: str, format: str = 'json'):
        """
        Export experiment data to file.

        Args:
            experiment_data (Dict[str, Any]): Experiment data
            output_path (str): Output file path
            format (str): Export format ('json', 'csv', 'excel')
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == 'json':
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._prepare_data_for_json(experiment_data)
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2, default=str)

            elif format.lower() == 'csv':
                # Create CSV comparison table
                run_list = list(experiment_data['runs'].values())
                df = self.compare_runs(run_list)
                df.to_csv(output_path, index=False)

            elif format.lower() == 'excel':
                # Export to Excel
                run_list = list(experiment_data['runs'].values())
                df = self.compare_runs(run_list)
                df.to_excel(output_path, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            print(f"✅ Experiment data exported to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to export data: {e}")

    def _prepare_data_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JSON serialization."""
        json_data = {}

        def convert_value(value):
            if hasattr(value, 'tolist'):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            else:
                return value

        for key, value in data.items():
            json_data[key] = convert_value(value)

        return json_data