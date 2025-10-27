"""
CNN Training Module for Protein Contact Prediction

This module provides a complete training pipeline for CNN-based protein contact prediction
with binary output format (0/1 contact maps) as required by the problem definition.

The implementation includes:
- Binary CNN architecture (no sigmoid activation)
- Dataset loading for tiny_10 synchronized dataset
- Memory-optimized training with mixed precision
- Comprehensive evaluation metrics
- Loss functions for binary classification

Key Features:
- Input: 68-channel tensors (4 template + 64 ESM2 channels)
- Output: Binary contact maps (0/1 values only)
- Memory: Optimized for batch size 1, FP16 training
- Evaluation: AUC, Precision@L, Precision@L5 metrics

Usage:
    from esm2_contact.training import BinaryContactCNN, Tiny10Dataset, CNNTrainer

    # Load dataset
    dataset = Tiny10Dataset("data/tiny_10/cnn_dataset.h5")

    # Create model
    model = BinaryContactCNN(in_channels=68, base_channels=32)

    # Train
    trainer = CNNTrainer(model, dataset)
    history, best_auc = trainer.train()
"""

from .model import BinaryContactCNN
from .dataset import Tiny10Dataset, collate_fn, create_data_splits
from .trainer import CNNTrainer
from .metrics import ContactMetrics
from .losses import get_loss_function, FocalLoss

__all__ = [
    'BinaryContactCNN',
    'Tiny10Dataset',
    'CNNTrainer',
    'ContactMetrics',
    'get_loss_function',
    'FocalLoss',
    'collate_fn',
    'create_data_splits'
]

__version__ = '1.0.0'