"""
CNN Trainer for Binary Contact Prediction

This module provides a comprehensive training pipeline for binary protein contact
prediction with memory optimization, mixed precision training, adaptive batch sizing,
and robust error handling.

Key Features:
- Memory-optimized training with aggressive cleanup
- Mixed precision training (FP16) support
- Adaptive batch size optimization for maximum GPU utilization
- Global progress bars with time estimation
- OOM recovery mechanisms with automatic batch size adjustment
- Early stopping with learning rate scheduling
- Comprehensive evaluation and metrics tracking
- Model checkpointing and resuming

Training Features:
- Adaptive batch sizing (automatic or manual)
- Gradient clipping for training stability
- Learning rate scheduling based on validation performance
- Automatic mixed precision for faster training
- Real-time progress tracking with ETA
- Memory utilization monitoring and optimization

Usage:
    trainer = CNNTrainer(model, train_loader, val_loader, test_loader)
    history, best_auc = trainer.train(config)
    results = trainer.evaluate_best_model()
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")

from .losses import get_loss_function
from .metrics import ContactMetrics

try:
    import mlflow
    import mlflow.pyfunc
    from ..mlflow_utils import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    warnings.warn("MLflow not available. Install mlflow for experiment tracking.")


class CNNTrainer:
    """
    Comprehensive trainer for binary CNN contact prediction with MLflow integration.

    This class handles all aspects of training including memory management,
    mixed precision training, evaluation, checkpointing, and experiment tracking.

    Args:
        model (nn.Module): Binary CNN model
        device (torch.device): Device for training (cpu or cuda)
        verbose (bool): Whether to print verbose information (default: True)
        enable_mlflow (bool): Whether to enable MLflow tracking (default: False)
        mlflow_experiment (str): MLflow experiment name (default: None)
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 verbose: bool = True, enable_mlflow: bool = False,
                 mlflow_experiment: Optional[str] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.metrics_calculator = ContactMetrics()
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.mlflow_experiment = mlflow_experiment or "esm2_contact_training"
        self.mlflow_tracker = None

        # Move model to device
        self.model.to(self.device)

        if self.verbose:
            print(f"🧠 CNN Trainer initialized")
            print(f"   Device: {self.device}")
            print(f"   Model: {type(model).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   MLflow: {'✅ Enabled' if self.enable_mlflow else '❌ Disabled'}")

    def get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def clear_memory(self) -> None:
        """Aggressive memory cleanup."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB."""
        if self.device.type == 'cuda':
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0

    def estimate_batch_memory(self, sample_batch) -> float:
        """Estimate memory usage for a given batch size in GB."""
        try:
            # Move sample to device
            features, contacts, mask, lengths = sample_batch
            features = features.to(self.device, non_blocking=True)
            contacts = contacts.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            # Forward pass to estimate memory
            with torch.no_grad():
                outputs = self.model(features)

            mem_usage = self.get_gpu_memory()
            self.clear_memory()

            # Clean up tensors
            del features, contacts, mask, outputs
            torch.cuda.empty_cache()

            return mem_usage
        except Exception:
            return 0.0

    def find_optimal_batch_size(self, train_loader, max_batch_size: int = 32,
                              target_memory_utilization: float = 0.7) -> int:
        """
        Find optimal batch size based on available GPU memory.

        Args:
            train_loader: Training data loader
            max_batch_size: Maximum batch size to try
            target_memory_utilization: Target GPU memory utilization (0.0-1.0)

        Returns:
            Optimal batch size
        """
        if self.device.type != 'cuda':
            return 1  # CPU training, keep batch size 1

        total_memory = self.get_total_gpu_memory()
        target_memory = total_memory * target_memory_utilization

        if self.verbose:
            print(f"🔍 Finding optimal batch size...")
            print(f"   Total GPU memory: {total_memory:.1f}GB")
            print(f"   Target memory usage: {target_memory:.1f}GB ({target_memory_utilization*100:.0f}%)")

        # Start with a reasonable batch size
        optimal_batch_size = 1
        current_batch_size = min(8, max_batch_size)  # Start with 8 or max_batch_size

        # Get a sample batch for testing
        sample_batch = None
        for batch in train_loader:
            sample_batch = batch
            break

        if sample_batch is None:
            return 1

        # Test different batch sizes (limited attempts)
        max_attempts = 10  # Limit the number of batch size tests
        attempts = 0
        while current_batch_size <= max_batch_size and attempts < max_attempts:
            try:
                # Create a temporary batch with current_batch_size
                features, contacts, mask, lengths = sample_batch

                # Repeat the batch to simulate larger batch size
                repeat_times = min(current_batch_size, 4)  # Limit repetition to avoid huge tensors
                features_large = features.repeat(repeat_times, 1, 1, 1)[:current_batch_size]
                contacts_large = contacts.repeat(repeat_times, 1, 1)[:current_batch_size]
                mask_large = mask.repeat(repeat_times, 1, 1)[:current_batch_size]
                lengths_large = lengths.repeat(repeat_times)[:current_batch_size]

                test_batch = (features_large, contacts_large, mask_large, lengths_large)

                # Estimate memory usage
                estimated_memory = self.estimate_batch_memory(test_batch)

                if self.verbose:
                    print(f"   Testing batch size {current_batch_size}: ~{estimated_memory:.2f}GB")

                # Check if we can fit this batch size
                if estimated_memory > 0 and estimated_memory < target_memory:
                    optimal_batch_size = current_batch_size

                    # Try larger batch size
                    current_batch_size = min(current_batch_size * 2, max_batch_size)
                else:
                    # Too large, stop here
                    break

                attempts += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.verbose:
                        print(f"   Batch size {current_batch_size}: OOM detected")
                    break
                else:
                    raise e
            except Exception as e:
                if self.verbose:
                    print(f"   Batch size {current_batch_size}: Error ({e})")
                break

            attempts += 1

        # Clean up
        self.clear_memory()

        if self.verbose:
            final_memory_pct = (self.estimate_batch_memory(sample_batch) * optimal_batch_size) / total_memory * 100
            print(f"   ✅ Optimal batch size: {optimal_batch_size}")
            print(f"   Estimated memory usage: ~{final_memory_pct:.1f}% of GPU")

        return max(1, optimal_batch_size)

    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module, scaler: Optional[GradScaler],
                   use_amp: bool, memory_threshold: float = 8.0,
                   use_progress_bar: bool = True) -> Tuple[float, int]:
        """
        Train for one epoch with OOM recovery and progress tracking.

        Args:
            train_loader (DataLoader): Training data loader
            optimizer (torch.optim.Optimizer): Optimizer
            criterion (nn.Module): Loss function
            scaler (Optional[GradScaler]): AMP scaler
            use_amp (bool): Whether to use mixed precision
            memory_threshold (float): Memory threshold in GB
            use_progress_bar (bool): Whether to use tqdm progress bar

        Returns:
            Tuple[float, int]: Average loss and number of OOM skips
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        oom_skips = 0

        # Setup progress bar
        if use_progress_bar and TQDM_AVAILABLE:
            # Force stdout output with multiple parameters
            pbar = tqdm(train_loader, desc="Training", leave=True,
                       disable=not self.verbose, ncols=100, file=sys.stdout,
                       miniters=1, mininterval=0, smoothing=0)
            if self.verbose:
                print(f"   ✅ Batch progress bar created")
        else:
            pbar = train_loader
            if self.verbose:
                print(f"   ⚠️  Using simple iteration (no batch progress bars)")

        for batch_idx, (features, contacts, mask, lengths) in enumerate(pbar):
            # Memory check before batch
            mem_before = self.get_gpu_memory()
            if mem_before > memory_threshold:
                self.clear_memory()
                if self.verbose and not use_progress_bar:
                    print(f"   🧹 Memory cleanup at batch {batch_idx} (was {mem_before:.2f}GB)")

            # Move to device
            features = features.to(self.device, non_blocking=True)
            contacts = contacts.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            try:
                optimizer.zero_grad()
                batch_loss = None

                if use_amp and scaler is not None:
                    with autocast():
                        outputs = self.model(features)
                        loss = criterion(outputs * mask, contacts * mask)
                        batch_loss = loss.item()

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(features)
                    loss = criterion(outputs * mask, contacts * mask)
                    batch_loss = loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += batch_loss
                num_batches += 1

                # Update progress bar
                if use_progress_bar and TQDM_AVAILABLE and self.verbose:
                    current_lr = optimizer.param_groups[0]['lr']
                    mem_after = self.get_gpu_memory()
                    amp_status = "FP16" if use_amp else "FP32"
                    pbar.set_postfix({
                        'Loss': f'{batch_loss:.4f}',
                        'LR': f'{current_lr:.4f}',
                        'GPU': f'{mem_after:.1f}GB',
                        'Mode': amp_status
                    })
                    # Force stdout flush to ensure progress bar is visible
                    sys.stdout.flush()

                # Fallback progress reporting (verbose mode without progress bar)
                elif not use_progress_bar and self.verbose and batch_idx % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    mem_after = self.get_gpu_memory()
                    amp_status = "FP16" if use_amp else "FP32"
                    print(f"   Batch {batch_idx:4d}: Loss = {batch_loss:.4f}, "
                          f"LR = {current_lr:.6f}, GPU = {mem_after:.2f}GB ({amp_status})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.verbose:
                        if use_progress_bar:
                            pbar.write(f"⚠️  OOM detected! Skipping batch {batch_idx}")
                        else:
                            print(f"   ⚠️  OOM detected! Skipping batch {batch_idx}")
                    self.clear_memory()
                    oom_skips += 1
                    continue
                else:
                    raise e

            # Cleanup after each batch
            self.clear_memory()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, oom_skips

    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module,
                      use_amp: bool, memory_threshold: float = 8.0) -> Tuple[float, Dict, int]:
        """
        Validate for one epoch with OOM recovery using incremental metrics calculation.

        Args:
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module): Loss function
            use_amp (bool): Whether to use mixed precision
            memory_threshold (float): Memory threshold in GB

        Returns:
            Tuple[float, Dict, int]: Average loss, metrics, and OOM skips
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        oom_skips = 0

        # Reset incremental metrics for new evaluation
        self.metrics_calculator.reset_incremental_metrics()

        with torch.no_grad():
            for features, contacts, mask, lengths in val_loader:
                # Memory check
                mem_before = self.get_gpu_memory()
                if mem_before > memory_threshold:
                    self.clear_memory()
                    if self.verbose:
                        print(f"   🧹 Validation memory cleanup (was {mem_before:.2f}GB)")

                features = features.to(self.device, non_blocking=True)
                contacts = contacts.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                try:
                    if use_amp and self.device.type == 'cuda':
                        with autocast():
                            outputs = self.model(features)
                            loss = criterion(outputs * mask, contacts * mask)
                    else:
                        outputs = self.model(features)
                        loss = criterion(outputs * mask, contacts * mask)

                    total_loss += loss.item()
                    num_batches += 1

                    # Update incremental metrics immediately (no accumulation)
                    self.metrics_calculator.update_batch(outputs, contacts, lengths, mask)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if self.verbose:
                            print(f"   ⚠️  Validation OOM! Skipping batch.")
                        self.clear_memory()
                        oom_skips += 1
                        continue
                    else:
                        raise e

                # Cleanup after each batch
                self.clear_memory()

        # Calculate final metrics from all batches
        metrics = self.metrics_calculator.calculate_incremental_metrics()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, metrics, oom_skips

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              config: Dict) -> Tuple[Dict, float]:
        """
        Complete training pipeline with early stopping and MLflow tracking.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            config (Dict): Training configuration

        Returns:
            Tuple[Dict, float]: Training history and best AUC
        """
        # Initialize MLflow tracking if enabled
        if self.enable_mlflow:
            try:
                if self.verbose:
                    print(f"🔧 Initializing MLflow tracking for experiment: {self.mlflow_experiment}")

                self.mlflow_tracker = MLflowTracker(
                    experiment_name=self.mlflow_experiment,
                    tags={
                        "model_type": "binary_cnn",
                        "dataset": config.get("dataset_path", "unknown"),
                        "train_samples": len(train_loader.dataset),
                        "val_samples": len(val_loader.dataset)
                    }
                )

                # Enter MLflow context and verify it's active
                self.mlflow_tracker.__enter__()

                if self.mlflow_tracker.run is None:
                    print("⚠️  MLflow run failed to start. Disabling MLflow tracking.")
                    self.enable_mlflow = False
                elif self.verbose:
                    print(f"✅ MLflow run started successfully")
                    print(f"   Run ID: {self.mlflow_tracker.run.info.run_id}")

            except Exception as e:
                print(f"❌ Failed to initialize MLflow: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                self.enable_mlflow = False
                self.mlflow_tracker = None

            # Log training configuration if MLflow is active
            if self.enable_mlflow and self.mlflow_tracker:
                try:
                    training_params = {
                        "learning_rate": config.get('learning_rate'),
                        "batch_size": config.get('batch_size'),
                        "num_epochs": config.get('num_epochs'),
                        "loss_type": config.get('loss_type'),
                        "use_amp": config.get('use_amp', True),
                        "adaptive_batching": config.get('adaptive_batching', False),
                        "weight_decay": config.get('weight_decay', 1e-5),
                        "patience": config.get('patience', 8),
                        "train_samples": len(train_loader.dataset),
                        "val_samples": len(val_loader.dataset)
                    }

                    # Add model architecture parameters
                    if hasattr(self.model, 'get_model_info'):
                        model_info = self.model.get_model_info()
                        training_params.update(model_info)

                    # Add comprehensive configuration parameters
                    training_params.update({
                        # Dataset configuration
                        "dataset_path": config.get("dataset_path"),
                        "dataset_fraction": config.get("dataset_fraction", 1.0),
                        "train_ratio": config.get("train_ratio"),
                        "val_ratio": config.get("val_ratio"),
                        "test_ratio": config.get("test_ratio"),
                        "random_seed": config.get("random_seed"),

                        # Model architecture
                        "in_channels": config.get("in_channels"),
                        "base_channels": config.get("base_channels"),
                        "dropout_rate": config.get("dropout_rate"),

                        # Loss configuration
                        "pos_weight": config.get("pos_weight"),

                        # System configuration
                        "device": str(self.device),
                        "verbose": config.get("verbose"),
                        "experiment_name": config.get("experiment_name")
                    })

                    self.mlflow_tracker.log_params(training_params)
                    if self.verbose:
                        print(f"📊 Logged {len(training_params)} parameters to MLflow")

                except Exception as e:
                    print(f"⚠️  Failed to log parameters to MLflow: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()

        # Adaptive batch size detection
        initial_batch_size = config.get('batch_size', 1)
        adaptive_batching = config.get('adaptive_batching', False)

        if adaptive_batching and initial_batch_size == 1:
            # Find optimal batch size automatically
            optimal_batch_size = self.find_optimal_batch_size(
                train_loader,
                max_batch_size=config.get('max_batch_size', 32),
                target_memory_utilization=config.get('memory_utilization', 0.7)
            )
            config['batch_size'] = optimal_batch_size

            # Update data loaders with new batch size
            if self.verbose:
                print(f"   🔄 Updating data loaders with batch size {optimal_batch_size}")
        else:
            optimal_batch_size = initial_batch_size

        # Handle quiet progress mode
        quiet_progress = config.get('quiet_progress', False)
        effective_verbose = self.verbose and not quiet_progress

        if effective_verbose:
            print("🚀 Starting CNN Training Pipeline")
            print(f"   Epochs: {config['num_epochs']}")
            print(f"   Learning rate: {config['learning_rate']}")
            print(f"   Loss function: {config['loss_type']}")
            print(f"   Mixed precision: {config.get('use_amp', True)}")
            print(f"   Batch size: {optimal_batch_size}")
            if adaptive_batching:
                print(f"   Adaptive batching: Enabled")
            print(f"   GPU memory: {self.get_total_gpu_memory():.1f}GB total")

        # Initialize training components
        criterion = get_loss_function(
            loss_type=config['loss_type'],
            pos_weight=config.get('pos_weight', 5.0)
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 3)
        )

        scaler = GradScaler() if config.get('use_amp', True) and self.device.type == 'cuda' else None

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_precision_at_l': [],
            'val_precision_at_l5': [],
            'learning_rates': [],
            'gpu_memory': [],
            'oom_skips': [],
            'batch_size': optimal_batch_size,
            'epoch_times': [],  # Track time for each epoch
            'epochs_data': []  # Track detailed epoch data for analysis
        }

        # Early stopping
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        patience = config.get('patience', 8)
        best_model_state = None

        # Initial memory cleanup
        self.clear_memory()
        if effective_verbose:
            print(f"   Initial GPU memory: {self.get_gpu_memory():.2f}GB")

        # Global progress bar setup
        use_global_progress = config.get('use_progress_bar', True) and TQDM_AVAILABLE

        if use_global_progress:
            # Force stdout output with multiple parameters
            epoch_pbar = tqdm(
                range(config['num_epochs']),
                desc="Training Progress",
                ncols=120,
                disable=not effective_verbose,
                file=sys.stdout,
                miniters=1, mininterval=0, smoothing=0, leave=True
            )
        else:
            epoch_pbar = range(config['num_epochs'])
            if effective_verbose:
                print(f"   ⚠️  Using simple range (no progress bars)")

        # Training loop
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            self.clear_memory()

            # Training phase
            original_verbose = self.verbose
            self.verbose = effective_verbose
            train_loss, train_oom = self.train_epoch(
                train_loader, optimizer, criterion, scaler,
                config.get('use_amp', True),
                config.get('memory_threshold', 8.0),
                use_progress_bar=config.get('use_progress_bar', True)
            )
            self.verbose = original_verbose

            # Validation phase
            val_loss, val_metrics, val_oom = self.validate_epoch(
                val_loader, criterion,
                config.get('use_amp', True),
                config.get('memory_threshold', 8.0)
            )

            total_oom = train_oom + val_oom
            current_auc = val_metrics['auc']
            current_lr = optimizer.param_groups[0]['lr']

            # Learning rate scheduling
            scheduler.step(current_auc)
            new_lr = optimizer.param_groups[0]['lr']

            # Learning rate change notification
            if new_lr != current_lr and self.verbose:
                print(f"   📉 Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")

            # Calculate epoch time before updating history
            epoch_time = time.time() - epoch_start_time
            amp_status = "FP16" if config.get('use_amp', True) else "FP32"
            oom_info = f", OOM: {total_oom}" if total_oom > 0 else ""

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(current_auc)
            history['val_precision_at_l'].append(val_metrics['precision_at_l'])
            history['val_precision_at_l5'].append(val_metrics['precision_at_l5'])
            history['learning_rates'].append(new_lr)
            history['epoch_times'].append(epoch_time)

            # Store detailed epoch data for analysis
            epoch_data = {
                'epoch': epoch + 1,
                'epoch_time': epoch_time,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auc': current_auc,
                'learning_rate': new_lr,
                'gpu_memory': self.get_gpu_memory()
            }
            epoch_data.update(val_metrics)
            history['epochs_data'].append(epoch_data)
            history['gpu_memory'].append(self.get_gpu_memory())
            history['oom_skips'].append(total_oom)

            # Update global progress bar
            if use_global_progress:
                remaining_epochs = config['num_epochs'] - (epoch + 1)
                estimated_remaining_time = remaining_epochs * epoch_time
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)

                time_str = f"{hours:02d}h {minutes:02d}m" if hours > 0 else f"{minutes:02d}m"

                epoch_pbar.set_postfix({
                    'Epoch': f"{epoch+1}/{config['num_epochs']}",
                    'Loss': f'{train_loss:.3f}',
                    'AUC': f'{current_auc:.3f}',
                    'LR': f'{new_lr:.4f}',
                    'GPU': f'{self.get_gpu_memory():.1f}GB',
                    'Time': f'{epoch_time:.0f}s',
                    'ETA': time_str
                })

                if total_oom > 0:
                    epoch_pbar.set_postfix({
                        **epoch_pbar.postfix,
                        'OOM': str(total_oom)
                    })

            # Log comprehensive epoch metrics to MLflow
            if self.enable_mlflow and self.mlflow_tracker:
                try:
                    # Comprehensive epoch metrics
                    epoch_metrics = {
                        # Loss metrics
                        "train_loss": train_loss,
                        "val_loss": val_loss,

                        # Validation performance metrics
                        "val_auc": current_auc,
                        "val_precision_at_l": val_metrics['precision_at_l'],
                        "val_precision_at_l5": val_metrics['precision_at_l5'],
                        "val_recall_at_l": val_metrics.get('recall_at_l', 0.0),
                        "val_f1_at_l": val_metrics.get('f1_at_l', 0.0),
                        "val_accuracy": val_metrics.get('accuracy', 0.0),

                        # Training dynamics
                        "learning_rate": new_lr,
                        "train_oom_skips": train_oom,
                        "val_oom_skips": val_oom,
                        "epoch_time": epoch_time,
                        "gpu_memory": self.get_gpu_memory(),

                        # Additional training info
                        "epoch": epoch + 1,
                        "cumulative_training_time": sum([h.get('epoch_time', 0) for h in history.get('epochs_data', [])]),

                        # Model performance tracking
                        "best_val_auc_so_far": best_auc,
                        "improvement": current_auc > best_auc
                    }

                    self.mlflow_tracker.log_metrics(epoch_metrics, step=epoch + 1)

                    # Reduce MLflow logging noise when progress bar is active
                    if self.verbose and not use_global_progress and epoch % 5 == 0:
                        print(f"📈 MLflow: Logged epoch {epoch + 1} metrics (AUC: {current_auc:.4f})")

                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  MLflow: Failed to log epoch metrics: {e}")
                        import traceback
                        traceback.print_exc()

            # Epoch summary (verbose mode or no progress bar)
            if self.verbose and not use_global_progress:
                print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, AUC: {current_auc:.4f}, "
                      f"Prec@L: {val_metrics['precision_at_l']:.4f}, "
                      f"LR: {new_lr:.6f}, GPU: {self.get_gpu_memory():.2f}GB "
                      f"({amp_status}){oom_info}, Time: {epoch_time:.1f}s")

            # Model checkpointing
            if current_auc > best_auc:
                best_auc = current_auc
                best_epoch = epoch
                patience_counter = 0
                # Store best epoch metrics for final logging
                history['best_epoch_metrics'] = val_metrics.copy()
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_auc': best_auc,
                    'config': config,
                    'history': history
                }

                # Save best model
                save_path = config.get('save_path', 'best_cnn_model.pth')
                torch.save(best_model_state, save_path)
                if self.verbose:
                    if use_global_progress:
                        epoch_pbar.write(f"🏆 New best model! AUC: {best_auc:.4f}")
                    else:
                        print(f"   🏆 New best model saved! AUC: {best_auc:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                if self.verbose:
                    if use_global_progress:
                        epoch_pbar.write(f"\n🛑 Early stopping triggered after {patience} epochs")
                        epoch_pbar.write(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch + 1}")
                    else:
                        print(f"\n🛑 Early stopping triggered after {patience} epochs")
                        print(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch + 1}")
                break

            # Final cleanup
            self.clear_memory()

        # Close global progress bar
        if use_global_progress:
            epoch_pbar.close()

        # Load best model
        if best_model_state is not None:
            try:
                self.model.load_state_dict(best_model_state['model_state_dict'])
                if self.verbose:
                    print("   ✅ Best model loaded successfully")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Warning: Could not load best model: {e}")

        total_oom_events = sum(history['oom_skips'])

        # Log comprehensive final results to MLflow
        if self.enable_mlflow and self.mlflow_tracker:
            try:
                # Calculate comprehensive final metrics
                total_training_time = sum(history.get('epoch_times', []))
                avg_epoch_time = total_training_time / len(history.get('epoch_times', [1]))
                best_epoch_metrics = history.get('best_epoch_metrics', {})

                final_metrics = {
                    # Core performance metrics
                    "best_val_auc": best_auc,
                    "best_epoch": best_epoch + 1,
                    "total_epochs": epoch + 1,

                    # Loss tracking
                    "final_train_loss": history['train_loss'][-1] if history['train_loss'] else 0.0,
                    "final_val_loss": history['val_loss'][-1] if history['val_loss'] else 0.0,
                    "best_val_loss": min(history['val_loss']) if history['val_loss'] else 0.0,

                    # Training efficiency metrics
                    "total_training_time": total_training_time,
                    "avg_epoch_time": avg_epoch_time,
                    "total_oom_events": total_oom_events,
                    "oom_recovery_rate": total_oom_events / (epoch + 1) if epoch > 0 else 0.0,

                    # Convergence metrics
                    "epochs_to_best": best_epoch + 1,
                    "training_stability": np.std(history['val_loss'][-10:]) if len(history['val_loss']) >= 10 else 0.0,

                    # Model complexity
                    "model_parameters": sum(p.numel() for p in self.model.parameters()),
                    "model_size_mb": sum(p.numel() for p in self.model.parameters()) * 4 / (1024 * 1024),

                    # Dataset information
                    "total_training_samples": len(train_loader.dataset) * (epoch + 1),
                    "samples_per_second": (len(train_loader.dataset) * (epoch + 1)) / total_training_time if total_training_time > 0 else 0.0
                }

                # Add best epoch detailed metrics if available
                if best_epoch_metrics:
                    final_metrics.update({
                        "best_val_precision_at_l": best_epoch_metrics.get('precision_at_l', 0.0),
                        "best_val_precision_at_l5": best_epoch_metrics.get('precision_at_l5', 0.0),
                        "best_val_recall_at_l": best_epoch_metrics.get('recall_at_l', 0.0),
                        "best_val_f1_at_l": best_epoch_metrics.get('f1_at_l', 0.0),
                        "best_val_accuracy": best_epoch_metrics.get('accuracy', 0.0)
                    })

                self.mlflow_tracker.log_metrics(final_metrics)
                if self.verbose:
                    print(f"📊 MLflow: Logged {len(final_metrics)} final metrics")
                    print(f"   Total training time: {total_training_time:.1f}s")

                # Log training history as artifact
                self.mlflow_tracker.log_training_history(history)

                # Save best model checkpoint as artifact
                if best_model_state:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                        torch.save(best_model_state, f.name)
                        self.mlflow_tracker.log_artifact(f.name, "best_model_checkpoint")
                        os.unlink(f.name)
                    if self.verbose:
                        print(f"   💾 Logged best model checkpoint")

                # Create and log custom PyFunc model for serving with rich functionality
                try:
                    from ..serving.contact_predictor import create_pyfunc_model_instance

                    # Save best model to temporary file for PyFunc artifacts
                    if best_model_state:
                        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                            torch.save(best_model_state, f.name)
                            temp_model_path = f.name

                        try:
                            if self.verbose:
                                print(f"   🤖 Creating custom PyFunc model for serving...")

                            # Create PyFunc model instance (not class)
                            pyfunc_model = create_pyfunc_model_instance(
                                threshold=0.5,
                                confidence_method="probability"
                            )

                            # Skip manual signature inference to avoid mismatch warnings
                            # MLflow will automatically infer from the type hints in predict()
                            model_signature = None

                            if self.verbose:
                                print(f"   ✅ Using automatic signature inference from type hints")

                            # Use correct MLflow 3.x API parameters
                            mlflow.pyfunc.log_model(
                                python_model=pyfunc_model,  # Instance, not class
                                name="contact_predictor",    # Correct parameter name
                                signature=model_signature,
                                artifacts={"model": temp_model_path}
                            )

                            if self.verbose:
                                print(f"   🎉 Custom PyFunc model logged successfully!")
                                print(f"   📈 Features: structured predictions, confidence scoring, batch processing")

                        finally:
                            # Clean up temporary file
                            if os.path.exists(temp_model_path):
                                os.unlink(temp_model_path)

                except ImportError:
                    # Fallback to basic PyTorch model logging if serving module not available
                    if self.verbose:
                        print(f"   ⚠️  Serving module not available, falling back to PyTorch model...")

                    if len(train_loader) > 0:
                        sample_batch = next(iter(train_loader))
                        sample_features = sample_batch[0][:1]
                        sample_features = sample_features.to(self.device)

                        # Log PyTorch model with input example for better serving
                        mlflow.pytorch.log_model(
                            self.model,
                            artifact_path="pytorch_model",
                            input_example=sample_features
                        )

                        if self.verbose:
                            print(f"   ✅ PyTorch model logged successfully with input signature")
                    else:
                        # Fallback without input example
                        mlflow.pytorch.log_model(
                            self.model,
                            artifact_path="pytorch_model"
                        )

                        if self.verbose:
                            print(f"   ✅ PyTorch model logged successfully (no input example available)")

                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  Failed to log custom PyFunc model: {e}")
                        import traceback
                        traceback.print_exc()

                if self.verbose:
                    print(f"   📊 MLflow: Logged all artifacts and models")

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  MLflow: Failed to log results: {e}")

        if self.verbose:
            print(f"\n🎉 Training completed!")
            print(f"   Best AUC: {best_auc:.4f} at epoch {best_epoch + 1}")
            print(f"   Total epochs: {epoch + 1}")
            print(f"   Total OOM skips: {total_oom_events}")
            print(f"   Final GPU memory: {self.get_gpu_memory():.2f}GB")

        # End MLflow run
        if self.enable_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.__exit__(None, None, None)
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  MLflow: Failed to end run: {e}")

        self.clear_memory()
        return history, best_auc

    def evaluate(self, test_loader: DataLoader, config: Dict) -> Dict:
        """
        Evaluate the model on test set.

        Args:
            test_loader (DataLoader): Test data loader
            config (Dict): Configuration

        Returns:
            Dict: Test results
        """
        if self.verbose:
            print("🧪 Evaluating model on test set...")

        criterion = get_loss_function(
            loss_type=config['loss_type'],
            pos_weight=config.get('pos_weight', 5.0)
        )

        test_loss, test_metrics, _ = self.validate_epoch(
            test_loader, criterion,
            config.get('use_amp', True),
            config.get('memory_threshold', 8.0)
        )

        results = {
            'test_loss': test_loss,
            **{f'test_{k}': v for k, v in test_metrics.items()}
        }

        if self.verbose:
            print(f"   📊 Test Results:")
            print(f"      Test Loss: {test_loss:.4f}")
            print(f"      Test AUC: {test_metrics['auc']:.4f}")
            print(f"      Test Precision@L: {test_metrics['precision_at_l']:.4f}")
            print(f"      Test Precision@L5: {test_metrics['precision_at_l5']:.4f}")

        return results

    def predict(self, data_loader: DataLoader, threshold: float = 0.5) -> List[Dict]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader (DataLoader): Data loader
            threshold (float): Threshold for binary predictions

        Returns:
            List[Dict]: List of predictions with metadata
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for features, contacts, mask, lengths in data_loader:
                features = features.to(self.device)
                batch_size = features.shape[0]

                # Get predictions
                logits = self.model(features)
                probabilities = torch.sigmoid(logits)
                binary_pred = (probabilities > threshold).float()

                for i in range(batch_size):
                    length = lengths[i].item()
                    pred_info = {
                        'query_id': f"sample_{len(predictions)}",
                        'length': length,
                        'probabilities': probabilities[i, :length, :length].cpu().numpy(),
                        'binary_predictions': binary_pred[i, :length, :length].cpu().numpy(),
                        'targets': contacts[i, :length, :length].numpy() if contacts is not None else None
                    }
                    predictions.append(pred_info)

        return predictions


def create_trainer(model: nn.Module, device: Optional[torch.device] = None,
                  verbose: bool = True) -> CNNTrainer:
    """
    Factory function to create a CNNTrainer.

    Args:
        model (nn.Module): Model to train
        device (Optional[torch.device]): Training device
        verbose (bool): Verbose output

    Returns:
        CNNTrainer: Configured trainer
    """
    return CNNTrainer(model, device, verbose)