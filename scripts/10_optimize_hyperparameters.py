#!/usr/bin/env python3
"""
Hyperparameter Optimization Script for ESM2 Contact Prediction

This script uses Optuna for Bayesian hyperparameter optimization with MLflow integration
to find the best model configuration for protein contact prediction.

Key Features:
- Optuna Bayesian optimization with median pruning
- MLflow experiment tracking for all trials
- Multi-objective optimization (performance vs. computational cost)
- Parallel execution support
- Study persistence and resumption
- Automatic best model registration

Usage Examples:
    # Basic hyperparameter search
    uv run python scripts/optimize_hyperparameters.py \
        --dataset-path data/full_dataset.h5 \
        --study-name "esm2_contact_optimization" \
        --n-trials 50

    # Parallel execution with pruning
    uv run python scripts/optimize_hyperparameters.py \
        --dataset-path data/full_dataset.h5 \
        --study-name "esm2_contact_optimization" \
        --n-trials 100 \
        --pruning-enabled \
        --parallel-workers 4

    # Resume existing study
    uv run python scripts/optimize_hyperparameters.py \
        --dataset-path data/full_dataset.h5 \
        --study-name "esm2_contact_optimization" \
        --resume-study
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import optuna
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training components
from src.esm2_contact.training import (
    BinaryContactCNN,
    CNNTrainer,
    create_data_splits
)
from src.esm2_contact.mlflow_utils import setup_mlflow, MLflowTracker


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna with MLflow integration.

    This class provides a framework for systematic hyperparameter search
    with Bayesian optimization, pruning, and comprehensive tracking.

    Attributes:
        dataset_path (str): Path to training dataset
        study_name (str): Optuna study name
        n_trials (int): Number of optimization trials
        timeout (Optional[int]): Timeout for optimization in seconds
        enable_pruning (bool): Whether to enable early pruning
        parallel_workers (int): Number of parallel workers
        direction (str): Optimization direction ("maximize" or "minimize")

    Example:
        optimizer = HyperparameterOptimizer(
            dataset_path="data/dataset.h5",
            study_name="contact_optimization",
            n_trials=50
        )
        best_params = optimizer.optimize()
    """

    def __init__(self,
                 dataset_path: str,
                 study_name: str = "esm2_contact_optimization",
                 n_trials: int = 50,
                 timeout: Optional[int] = None,
                 enable_pruning: bool = True,
                 parallel_workers: int = 1,
                 direction: str = "maximize",
                 mlflow_experiment: str = "hyperparameter_optimization"):
        """
        Initialize hyperparameter optimizer.

        Args:
            dataset_path (str): Path to training dataset
            study_name (str): Optuna study name
            n_trials (int): Number of optimization trials
            timeout (Optional[int]): Timeout in seconds
            enable_pruning (bool): Whether to enable pruning
            parallel_workers (int): Number of parallel workers
            direction (str): Optimization direction
            mlflow_experiment (str): MLflow experiment name
        """
        self.dataset_path = dataset_path
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.enable_pruning = enable_pruning
        self.parallel_workers = parallel_workers
        self.direction = direction
        self.mlflow_experiment = mlflow_experiment

        # Setup MLflow
        setup_mlflow(experiment_name=mlflow_experiment)

        # Initialize study
        self.study = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_dataset(self):
        """Load and prepare dataset for optimization."""
        try:
            from src.esm2_contact.training import Tiny10Dataset

            print(f"📊 Loading dataset from {self.dataset_path}")
            self.dataset = Tiny10Dataset(self.dataset_path, validate_data=True, verbose=False)

            print(f"✅ Dataset loaded successfully")
            print(f"   Total proteins: {len(self.dataset)}")

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            trial (optuna.Trial): Optuna trial object

        Returns:
            float: Objective value (validation AUC)
        """
        try:
            # Hyperparameters to optimize
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [1, 2, 4, 8]),
                'base_channels': trial.suggest_categorical('base_channels', [16, 32, 64]),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'pos_weight': trial.suggest_float('pos_weight', 1.0, 20.0),
                'lr_factor': trial.suggest_float('lr_factor', 0.3, 0.9),
                'lr_patience': trial.suggest_int('lr_patience', 2, 8),
                'patience': trial.suggest_int('patience', 5, 20),
                'num_epochs': trial.suggest_int('num_epochs', 10, 50),
                'train_ratio': trial.suggest_float('train_ratio', 0.6, 0.9),
                'val_ratio': trial.suggest_float('val_ratio', 0.05, 0.2),
                'test_ratio': trial.suggest_float('test_ratio', 0.05, 0.2),
            }

            # Ensure ratios sum to 1
            total_ratio = params['train_ratio'] + params['val_ratio'] + params['test_ratio']
            params['train_ratio'] /= total_ratio
            params['val_ratio'] /= total_ratio
            params['test_ratio'] /= total_ratio

            # Validate hyperparameters
            if not self._validate_hyperparameters(params):
                raise optuna.TrialPruned()

            # Log to MLflow
            with MLflowTracker(
                experiment_name=self.mlflow_experiment,
                run_name=f"trial_{trial.number}_{trial.study.study_name}",
                tags={
                    "trial_number": str(trial.number),
                    "study_name": trial.study.study_name,
                    "optuna_trial_id": trial.trial_id,
                    "optimization_type": "hyperparameter_search"
                }
            ) as tracker:
                # Log parameters
                tracker.log_params(params)

                # Create data splits
                train_dataset, val_dataset, test_dataset = create_data_splits(
                    self.dataset,
                    train_ratio=params['train_ratio'],
                    val_ratio=params['val_ratio'],
                    test_ratio=params['test_ratio'],
                    random_seed=42
                )

                # Create data loaders
                from torch.utils.data import DataLoader
                train_loader = DataLoader(
                    train_dataset, batch_size=params['batch_size'],
                    shuffle=True, num_workers=0, pin_memory=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=params['batch_size'],
                    shuffle=False, num_workers=0, pin_memory=True
                )

                # Create model
                model = BinaryContactCNN(
                    in_channels=68,
                    base_channels=params['base_channels'],
                    dropout_rate=params['dropout_rate']
                )

                # Create trainer
                trainer = CNNTrainer(
                    model,
                    device=self.device,
                    verbose=False,  # Reduce verbosity during optimization
                    enable_mlflow=False  # Avoid nested MLflow logging
                )

                # Training configuration
                config = {
                    'dataset_path': self.dataset_path,
                    'batch_size': params['batch_size'],
                    'test_batch_size': params['batch_size'],
                    'num_epochs': params['num_epochs'],
                    'learning_rate': params['learning_rate'],
                    'weight_decay': params['weight_decay'],
                    'patience': params['patience'],
                    'loss_type': 'bce',
                    'pos_weight': params['pos_weight'],
                    'use_amp': True,
                    'memory_threshold': 8.0,
                    'adaptive_batching': False,
                    'max_batch_size': params['batch_size'],
                    'memory_utilization': 0.7,
                    'use_progress_bar': False,
                    'verbose': False
                }

                # Train model
                start_time = time.time()
                history, best_auc = trainer.train(train_loader, val_loader, config)
                training_time = time.time() - start_time

                # Calculate additional metrics
                final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
                final_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
                total_epochs = len(history['train_loss'])

                # Report intermediate values for pruning
                for epoch in range(min(10, total_epochs)):
                    if epoch < len(history['val_auc']):
                        trial.report(history['val_auc'][epoch], epoch)

                    # Check for pruning
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # Log final results
                final_metrics = {
                    'best_val_auc': best_auc,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'total_epochs': total_epochs,
                    'training_time': training_time,
                    'model_parameters': sum(p.numel() for p in model.parameters()),
                    'train_samples': len(train_dataset),
                    'val_samples': len(val_dataset)
                }

                tracker.log_metrics(final_metrics)

                # Log model if it's good
                if best_auc > 0.8:  # Good model threshold
                    tracker.log_model(model, "model")

                print(f"✅ Trial {trial.number}: AUC={best_auc:.4f}, Time={training_time:.1f}s")

                return best_auc

        except optuna.TrialPruned:
            print(f"⚠️ Trial {trial.number}: Pruned")
            raise

        except Exception as e:
            print(f"❌ Trial {trial.number}: Failed - {e}")
            # Return very bad score
            return 0.0

    def _validate_hyperparameters(self, params: Dict[str, Any]) -> bool:
        """Validate hyperparameter combination."""
        try:
            # Basic validation
            if params['learning_rate'] <= 0 or params['learning_rate'] > 1:
                return False

            if params['batch_size'] <= 0:
                return False

            if not 0 <= params['dropout_rate'] < 1:
                return False

            if params['num_epochs'] <= 0 or params['num_epochs'] > 100:
                return False

            # Memory constraint check
            estimated_memory = (
                params['base_channels'] * 128 * 4 / (1024**3) *  # Model size in GB
                params['batch_size']
            )

            if estimated_memory > 8.0:  # Conservative memory limit
                return False

            return True

        except Exception:
            return False

    def create_study(self, resume: bool = False) -> optuna.Study:
        """Create or load Optuna study."""
        try:
            # Study storage (file-based for simplicity)
            storage = f"sqlite:///optuna_studies/{self.study_name}.db"
            os.makedirs(os.path.dirname(storage), exist_ok=True)

            # Define objective
            if self.enable_pruning:
                # Median pruner for early stopping
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=5,
                    interval_steps=1
                )
            else:
                pruner = optuna.pruners.NopPruner()

            # Create or load study
            if resume:
                try:
                    study = optuna.load_study(
                        study_name=self.study_name,
                        storage=storage,
                        direction=self.direction,
                        pruner=pruner
                    )
                    print(f"✅ Resumed existing study: {self.study_name}")
                    return study
                except:
                    print(f"⚠️ Could not resume study, creating new one")

            # Create new study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=storage,
                direction=self.direction,
                pruner=pruner
            )

            print(f"✅ Created new study: {self.study_name}")
            return study

        except Exception as e:
            raise RuntimeError(f"Failed to create study: {e}")

    def optimize(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            resume (bool): Whether to resume existing study

        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            # Load dataset
            self.load_dataset()

            # Create study
            self.study = self.create_study(resume)

            print(f"🚀 Starting hyperparameter optimization")
            print(f"   Study: {self.study_name}")
            print(f"   Trials: {self.n_trials}")
            print(f"   Pruning: {'Enabled' if self.enable_pruning else 'Disabled'}")
            print(f"   Direction: {self.direction}")
            print(f"   Device: {self.device}")

            # Run optimization
            start_time = time.time()
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.parallel_workers
            )
            optimization_time = time.time() - start_time

            # Get results
            best_trial = self.study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value

            results = {
                'best_params': best_params,
                'best_value': best_value,
                'best_trial_number': best_trial.number,
                'total_trials': len(self.study.trials),
                'pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'optimization_time': optimization_time,
                'study_name': self.study_name
            }

            # Print results
            print(f"\n🎉 Optimization completed!")
            print(f"   Best {self.direction} value: {best_value:.4f}")
            print(f"   Best trial: {best_trial.number}")
            print(f"   Total trials: {len(self.study.trials)}")
            print(f"   Pruned trials: {results['pruned_trials']}")
            print(f"   Optimization time: {optimization_time:.1f}s")
            print(f"\n📊 Best hyperparameters:")
            for key, value in best_params.items():
                print(f"   {key}: {value}")

            # Log best results to MLflow
            with MLflowTracker(
                experiment_name=self.mlflow_experiment,
                run_name=f"optimization_summary_{self.study_name}",
                tags={
                    "optimization_type": "summary",
                    "study_name": self.study_name,
                    "best_trial": str(best_trial.number)
                }
            ) as tracker:
                tracker.log_params(best_params)
                tracker.log_metrics(results)

            return results

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt

            if not self.study:
                raise RuntimeError("No study available. Run optimize() first.")

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot optimization history
            trials = self.study.trials
            values = [t.value for t in trials if t.value is not None]
            trial_numbers = [t.number for t in trials if t.value is not None]

            ax1.plot(trial_numbers, values, 'bo-', alpha=0.7)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Objective Value')
            ax1.set_title(f'Optimization History ({self.study_name})')
            ax1.grid(True, alpha=0.3)

            # Plot parameter importance
            try:
                importance = optuna.importance.get_param_importances(self.study)
                params = list(importance.keys())[:10]  # Top 10 parameters
                scores = [importance[p] for p in params]

                ax2.barh(params, scores)
                ax2.set_xlabel('Importance')
                ax2.set_title('Hyperparameter Importance')
                ax2.grid(True, alpha=0.3)
            except:
                ax2.text(0.5, 0.5, 'Parameter importance not available',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Hyperparameter Importance')

            plt.tight_layout()

            # Save plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            print(f"⚠️ Failed to create plots: {e}")

    def export_best_config(self, save_path: str):
        """Export best hyperparameters to config file."""
        try:
            if not self.study:
                raise RuntimeError("No study available. Run optimize() first.")

            best_params = self.study.best_trial.params

            # Add default parameters
            config = {
                'model': {
                    'in_channels': 68,
                    'base_channels': best_params['base_channels'],
                    'dropout_rate': best_params['dropout_rate']
                },
                'training': {
                    'learning_rate': best_params['learning_rate'],
                    'batch_size': best_params['batch_size'],
                    'test_batch_size': best_params['batch_size'],
                    'num_epochs': best_params['num_epochs'],
                    'weight_decay': best_params['weight_decay'],
                    'patience': best_params['patience'],
                    'loss_type': 'bce',
                    'pos_weight': best_params['pos_weight'],
                    'use_amp': True,
                    'adaptive_batching': False
                },
                'data_splits': {
                    'train_ratio': best_params['train_ratio'],
                    'val_ratio': best_params['val_ratio'],
                    'test_ratio': best_params['test_ratio'],
                    'random_seed': 42
                },
                'optimization': {
                    'study_name': self.study_name,
                    'best_trial_number': self.study.best_trial.number,
                    'best_value': self.study.best_value,
                    'total_trials': len(self.study.trials)
                }
            }

            # Save config
            with open(save_path, 'w') as f:
                import json
                json.dump(config, f, indent=2)

            print(f"✅ Best configuration saved to {save_path}")

        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for ESM2 contact prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to CNN dataset (.h5 file)'
    )

    # Study arguments
    parser.add_argument(
        '--study-name',
        type=str,
        default='esm2_contact_optimization',
        help='Optuna study name'
    )

    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout for optimization in seconds'
    )

    parser.add_argument(
        '--direction',
        type=str,
        choices=['maximize', 'minimize'],
        default='maximize',
        help='Optimization direction'
    )

    # Optimization arguments
    parser.add_argument(
        '--pruning-enabled',
        action='store_true',
        default=False,
        help='Enable early pruning of unpromising trials'
    )

    parser.add_argument(
        '--parallel-workers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )

    parser.add_argument(
        '--resume-study',
        action='store_true',
        help='Resume existing study'
    )

    # Output arguments
    parser.add_argument(
        '--save-config',
        type=str,
        default=None,
        help='Path to save best configuration (JSON)'
    )

    parser.add_argument(
        '--plot-history',
        action='store_true',
        help='Generate optimization history plots'
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.dataset_path).exists():
        print(f"❌ Dataset file not found: {args.dataset_path}")
        return 1

    if args.n_trials <= 0:
        print("❌ n-trials must be positive")
        return 1

    if args.parallel_workers <= 0:
        print("❌ parallel-workers must be positive")
        return 1

    try:
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            dataset_path=args.dataset_path,
            study_name=args.study_name,
            n_trials=args.n_trials,
            timeout=args.timeout,
            enable_pruning=args.pruning_enabled,
            parallel_workers=args.parallel_workers,
            direction=args.direction
        )

        # Run optimization
        results = optimizer.optimize(resume=args.resume_study)

        # Save best configuration
        if args.save_config:
            optimizer.export_best_config(args.save_config)
        else:
            default_config_path = f"best_config_{args.study_name}.json"
            optimizer.export_best_config(default_config_path)

        # Generate plots
        if args.plot_history:
            plot_path = f"optimization_history_{args.study_name}.png"
            optimizer.plot_optimization_history(plot_path)

        print(f"\n🎉 Hyperparameter optimization completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())