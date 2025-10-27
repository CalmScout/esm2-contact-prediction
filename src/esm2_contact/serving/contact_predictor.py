"""
MLflow PyFunc Model for ESM2 Contact Prediction

This module provides a custom MLflow Python Function (PyFunc) for serving
ESM2-based protein contact prediction models following 2025 best practices.

Key Features:
- "Models from Code" approach for simplified implementation
- Input validation and preprocessing
- Batch processing support
- Confidence scoring and uncertainty quantification
- Structured input/output schemas
- Error handling and robustness

Usage:
    from esm2_contact.serving.contact_predictor import ContactPredictor

    # Create PyFunc model
    model = ContactPredictor(model_path="path/to/model.pth")

    # Predict contacts
    predictions = model.predict(features)

    # Save to MLflow
    mlflow.pyfunc.log_model(model, "contact_predictor")
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema

# Import project components
try:
    from ..training.model import BinaryContactCNN
    from ..training.metrics import ContactMetrics
except ImportError:
    warnings.warn("Could not import training components. Model loading may fail.")


class ContactPredictor:
    """
    Custom MLflow PyFunc for ESM2 contact prediction.

    This class implements the "Models from Code" approach for MLflow model serving,
    providing a clean interface for protein contact prediction with proper input validation
    and structured outputs.

    Attributes:
        model (nn.Module): The loaded PyTorch model
        device (torch.device): Device for inference
        threshold (float): Threshold for binary prediction
        confidence_method (str): Method for confidence scoring

    Example:
        # Create and save PyFunc
        model = ContactPredictor(model_path="model.pth")
        with mlflow.start_run():
            mlflow.pyfunc.log_model(model, "contact_model")

        # Load and use
        loaded_model = mlflow.pyfunc.load_model("runs:/.../contact_model")
        predictions = loaded_model.predict(features)
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 model_state: Optional[Dict] = None,
                 threshold: float = 0.5,
                 confidence_method: str = "probability",
                 device: Optional[str] = None):
        """
        Initialize ContactPredictor.

        Args:
            model_path (Optional[str]): Path to saved model checkpoint
            model_state (Optional[Dict]): Model state dictionary (alternative to path)
            threshold (float): Threshold for binary predictions (default: 0.5)
            confidence_method (str): Method for confidence scoring
                                    ("probability" or "margin", default: "probability")
            device (Optional[str]): Device for inference ("cpu", "cuda", or auto)
        """
        self.threshold = threshold
        self.confidence_method = confidence_method
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.metrics_calculator = ContactMetrics()

        # Load model
        if model_path:
            self._load_model_from_path(model_path)
        elif model_state:
            self._load_model_from_state(model_state)
        else:
            warnings.warn("No model provided. Use load_model() to load a model later.")

    def _load_model_from_path(self, model_path: str):
        """Load model from checkpoint file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint
                model_state = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})

                # Extract model architecture from config
                in_channels = config.get('in_channels', 68)
                base_channels = config.get('base_channels', 32)
                dropout_rate = config.get('dropout_rate', 0.1)

            else:
                # Just state dict
                model_state = checkpoint

                # Try to infer architecture from model state
                # This is a simplified inference - in production, save config separately
                in_channels = model_state.get('conv_blocks.0.0.weight', torch.zeros(1)).shape[1]
                base_channels = model_state.get('conv_blocks.0.3.weight', torch.zeros(1)).shape[0] // 2
                dropout_rate = 0.1

            # Create and load model
            self.model = BinaryContactCNN(
                in_channels=in_channels,
                base_channels=base_channels,
                dropout_rate=dropout_rate
            )
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()

            print(f"✅ Model loaded from {model_path}")
            print(f"   Architecture: {in_channels}→{base_channels} channels")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def _load_model_from_state(self, model_state: Dict):
        """Load model from state dictionary."""
        try:
            # Extract architecture info from model state
            in_channels = 68  # Default for ESM2 contact prediction
            base_channels = 32
            dropout_rate = 0.1

            self.model = BinaryContactCNN(
                in_channels=in_channels,
                base_channels=base_channels,
                dropout_rate=dropout_rate
            )
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()

            print("✅ Model loaded from state dictionary")
            print(f"   Architecture: {in_channels}→{base_channels} channels")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from state: {e}")

    def predict(self,
                context: mlflow.pyfunc.PythonModelContext,
                model_input: Union[np.ndarray, List[Dict[str, Any]], torch.Tensor]) -> Dict[str, Any]:
        """
        Predict protein contacts from input features.

        Args:
            context: MLflow model context
            model_input: Input features (numpy array, list of dicts, or torch tensor)

        Returns:
            Dict[str, Any]: Prediction results including binary contacts and confidence scores
        """
        return self._predict_batch(model_input)

    def _predict_batch(self,
                       model_input: Union[np.ndarray, List[Dict[str, Any]], torch.Tensor]) -> Dict[str, Any]:
        """
        Batch prediction with proper input handling.

        Args:
            model_input: Input features in various formats

        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Use load_model() first.")

        try:
            # Convert input to tensor
            features_tensor = self._prepare_input(model_input)

            # Move to device
            features_tensor = features_tensor.to(self.device)

            # Perform inference
            with torch.no_grad():
                logits = self.model(features_tensor)
                probabilities = torch.sigmoid(logits)
                binary_predictions = (probabilities > self.threshold).float()

                # Calculate confidence scores
                confidence_scores = self._calculate_confidence(probabilities, logits)

            # Convert to numpy for output
            predictions_np = binary_predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            confidence_np = confidence_scores.cpu().numpy()

            # Prepare results
            results = {
                "predictions": predictions_np.tolist(),
                "probabilities": probabilities_np.tolist(),
                "confidence_scores": confidence_np.tolist(),
                "threshold": self.threshold,
                "batch_size": len(predictions_np),
                "model_info": {
                    "input_shape": list(features_tensor.shape),
                    "device": str(self.device),
                    "confidence_method": self.confidence_method
                }
            }

            return results

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def _prepare_input(self, model_input: Union[np.ndarray, List[Dict[str, Any]], torch.Tensor]) -> torch.Tensor:
        """Convert various input formats to PyTorch tensor."""
        if isinstance(model_input, torch.Tensor):
            return model_input

        elif isinstance(model_input, np.ndarray):
            return torch.from_numpy(model_input).float()

        elif isinstance(model_input, list):
            # Handle list of feature dictionaries
            if all(isinstance(item, dict) for item in model_input):
                # Extract features from dictionaries
                features_list = []
                for item in model_input:
                    if 'features' in item:
                        features_list.append(torch.from_numpy(item['features']))
                    else:
                        # Assume the dict itself contains the feature array
                        features_list.append(torch.from_numpy(np.array(item)))

                if features_list:
                    return torch.stack(features_list).float()
                else:
                    raise ValueError("No 'features' key found in input dictionaries")
            else:
                # Assume list of numpy arrays
                return torch.stack([torch.from_numpy(arr) for arr in model_input]).float()

        else:
            raise ValueError(f"Unsupported input type: {type(model_input)}")

    def _calculate_confidence(self, probabilities: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence scores for predictions.

        Args:
            probabilities (torch.Tensor): Prediction probabilities
            logits (torch.Tensor): Raw model logits

        Returns:
            torch.Tensor: Confidence scores
        """
        if self.confidence_method == "probability":
            # Use probability as confidence
            return torch.max(probabilities, probabilities)  # Symmetric confidence

        elif self.confidence_method == "margin":
            # Use margin from threshold as confidence
            margin = torch.abs(probabilities - self.threshold)
            return margin

        else:
            warnings.warn(f"Unknown confidence method: {self.confidence_method}, using 'probability'")
            return probabilities

    def predict_single(self, features: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Predict contacts for a single protein.

        Args:
            features: Input features for one protein

        Returns:
            Dict[str, Any]: Prediction results for single protein
        """
        # Ensure input has batch dimension
        if len(features.shape) == 3:  # (channels, H, W)
            features = features.unsqueeze(0)  # Add batch dimension

        results = self._predict_batch(features)

        # Remove batch dimension for single prediction
        results['predictions'] = results['predictions'][0]
        results['probabilities'] = results['probabilities'][0]
        results['confidence_scores'] = results['confidence_scores'][0]
        results['batch_size'] = 1

        return results

    def evaluate_contacts(self,
                         features: torch.Tensor,
                         true_contacts: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate predictions against true contacts.

        Args:
            features: Input features
            true_contacts: True contact maps
            mask: Optional mask for valid regions

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Use load_model() first.")

        try:
            # Predict
            results = self._predict_batch(features)
            predictions = torch.tensor(results['predictions']).to(self.device)

            # Calculate lengths
            if mask is None:
                lengths = torch.tensor([pred.shape[-1] for pred in predictions]).to(self.device)
            else:
                lengths = torch.tensor([mask[i].sum().item() for i in range(len(mask))]).to(self.device)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(
                predictions, true_contacts, lengths, mask
            )

            return metrics

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")

    @staticmethod
    def get_input_schema() -> Schema:
        """Get input schema for MLflow model."""
        return Schema([
            ColSpec("features", DataType.float),
        ])

    @staticmethod
    def get_output_schema() -> Schema:
        """Get output schema for MLflow model."""
        return Schema([
            ColSpec("predictions", DataType.string),  # JSON string
            ColSpec("probabilities", DataType.string),  # JSON string
            ColSpec("confidence_scores", DataType.string),  # JSON string
            ColSpec("threshold", DataType.float),
            ColSpec("batch_size", DataType.integer),
        ])


def create_pyfunc_model(model_path: str,
                       signature: Optional[ModelSignature] = None,
                       **kwargs) -> Type[mlflow.pyfunc.PythonModel]:
    """
    Create MLflow PyFunc model from trained checkpoint.

    Args:
        model_path (str): Path to model checkpoint
        signature (Optional[ModelSignature]): Model signature
        **kwargs: Additional arguments for ContactPredictor

    Returns:
        Type[mlflow.pyfunc.PythonModel]: MLflow PyFunc model class
    """
    # Create ContactPredictor instance (for model loading)
    # This will be recreated in load_context
    predictor = ContactPredictor(model_path=model_path, **kwargs)

    # Create PythonModel class with closure over predictor and kwargs
    class ContactPredictionModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            """Load model artifacts."""
            actual_model_path = context.artifacts.get('model', model_path)
            self.predictor = ContactPredictor(model_path=actual_model_path, **kwargs)

        def predict(self,
                     context: Optional[mlflow.pyfunc.PythonModelContext],
                     model_input: list[Union[np.ndarray, List[Dict[str, Any]], torch.Tensor]]) -> Dict[str, Any]:
            """Make predictions with proper type hints.

            Args:
                context: MLflow model context (unused in this implementation)
                model_input: List of input features for prediction (MLflow expects list format)

            Returns:
                Dict[str, Any]: Prediction results
            """
            # Ensure predictor is loaded
            if not hasattr(self, 'predictor'):
                raise RuntimeError("Model not loaded. Call load_context() first.")

            # MLflow passes inputs as a list, handle both single and batch inputs
            if isinstance(model_input, list) and len(model_input) == 1:
                # Single input case - extract from list
                input_data = model_input[0]
            else:
                # Multiple inputs or already processed format
                input_data = model_input

            return self.predictor._predict_batch(input_data)

    return ContactPredictionModel


def log_model_to_mlflow(model_path: str,
                       artifact_path: str = "contact_model",
                       signature: Optional[ModelSignature] = None,
                       **kwargs):
    """
    Log contact prediction model to MLflow.

    Args:
        model_path (str): Path to model checkpoint
        artifact_path (str): MLflow artifact path
        signature (Optional[ModelSignature]): Model signature
        **kwargs: Additional arguments for ContactPredictor
    """
    # Create model
    pyfunc_model = create_pyfunc_model(model_path, signature, **kwargs)

    # Create conda environment
    conda_env = {
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python=3.10',
            'pytorch>=2.0.0',
            'torchvision>=0.15.0',
            'numpy>=1.20.0',
            'mlflow>=2.15.0',
            'scikit-learn>=1.0.0',
            'pandas>=1.3.0',
            'tqdm>=4.60.0',
            {
                'pip': [
                    'pip>=23.0.0',
                    'setuptools>=65.0.0',
                    'wheel>=0.40.0'
                ]
            }
        ],
        'name': 'esm2_contact_env'
    }

    # Log to MLflow using correct API syntax
    # The first parameter should be the model class (not instance)
    try:
        mlflow.pyfunc.log_model(
            pyfunc_model,
            artifact_path=artifact_path,
            signature=signature,
            artifacts={'model': model_path},
            conda_env=conda_env
        )
    except TypeError as e:
        if "artifact_path" in str(e):
            # Fallback: try without artifact_path parameter (newer MLflow versions)
            mlflow.pyfunc.log_model(
                pyfunc_model,
                signature=signature,
                artifacts={'model': model_path},
                conda_env=conda_env
            )
        else:
            raise e

    print(f"✅ Model logged to MLflow as '{artifact_path}'")