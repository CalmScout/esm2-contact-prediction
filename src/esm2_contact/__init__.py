"""
ESM2 Contact Prediction Package with MLflow Integration

This package provides utilities and models for protein contact prediction
using ESM2 embeddings and structural data from similar proteins, with
comprehensive MLflow support for experiment tracking and model serving.

Main Components:
- Dataset processing and generation
- Model training and evaluation with MLflow tracking
- Hyperparameter optimization with Optuna
- Model serving and deployment with MLflow PyFunc
"""

__version__ = "2.0.0"

# Core components
from .dataset import ContactDataset, collate_contact_maps, load_dataset_info

# Training components
from .training import BinaryContactCNN, CNNTrainer, Tiny10Dataset

# MLflow utilities
try:
    from .mlflow_utils import setup_mlflow, MLflowTracker, enable_pytorch_autolog
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

# Serving components
try:
    from .serving import ContactPredictor
    _SERVING_AVAILABLE = True
except ImportError:
    _SERVING_AVAILABLE = False

__all__ = [
    # Core components
    "ContactDataset",
    "collate_contact_maps",
    "load_dataset_info",

    # Training components
    "BinaryContactCNN",
    "CNNTrainer",
    "Tiny10Dataset",
]

# Add MLflow components if available
if _MLFLOW_AVAILABLE:
    __all__.extend([
        "setup_mlflow",
        "MLflowTracker",
        "enable_pytorch_autolog"
    ])

# Add serving components if available
if _SERVING_AVAILABLE:
    __all__.append("ContactPredictor")