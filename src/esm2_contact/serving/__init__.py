"""
Model Serving Module for ESM2 Contact Prediction

This module provides MLflow PyFunc models and utilities for serving
ESM2-based protein contact prediction models in production.

Components:
- ContactPredictor: Custom MLflow PyFunc for contact prediction
- Model serving utilities and helpers
- Batch inference interfaces

Usage:
    from esm2_contact.serving import ContactPredictor, log_model_to_mlflow

    # Create predictor
    model = ContactPredictor(model_path="model.pth")

    # Log to MLflow
    log_model_to_mlflow("model.pth", "contact_model")
"""

from .contact_predictor import ContactPredictor, create_pyfunc_model, log_model_to_mlflow

__all__ = [
    'ContactPredictor',
    'create_pyfunc_model',
    'log_model_to_mlflow'
]

__version__ = '1.0.0'