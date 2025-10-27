#!/usr/bin/env python3
"""
Batch Prediction Script for ESM2 Contact Prediction

This script provides a simple interface for batch inference using trained models
loaded from MLflow or local checkpoints.

Usage:
    # Predict with MLflow model
    uv run python scripts/batch_predict.py \
        --model-uri "models:/esm2_contact/Production" \
        --input-data data/test_features.h5 \
        --output predictions.json

    # Predict with local checkpoint
    uv run python scripts/batch_predict.py \
        --model-path experiments/best_model/model.pth \
        --input-data data/test_features.h5 \
        --output predictions.json
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import mlflow
import mlflow.pyfunc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.esm2_contact.serving.contact_predictor import ContactPredictor
except ImportError:
    warnings.warn("Could not import ContactPredictor. Using MLflow only.")


def load_input_data(input_path: str) -> np.ndarray:
    """
    Load input data from various formats.

    Args:
        input_path (str): Path to input file

    Returns:
        np.ndarray: Input features
    """
    try:
        print(f"📊 Loading input data from {input_path}")

        if input_path.endswith('.h5'):
            import h5py
            with h5py.File(input_path, 'r') as f:
                if 'features' in f:
                    features = f['features'][:]
                else:
                    # Try to find feature dataset
                    datasets = list(f.keys())
                    if datasets:
                        features = f[datasets[0]][:]
                    else:
                        raise ValueError("No data found in HDF5 file")

        elif input_path.endswith('.npy'):
            features = np.load(input_path)

        elif input_path.endswith('.json'):
            with open(input_path, 'r') as f:
                data = json.load(f)
                if 'features' in data:
                    features = np.array(data['features'])
                else:
                    features = np.array(data)

        elif input_path.endswith('.pt') or input_path.endswith('.pth'):
            features = torch.load(input_path, map_location='cpu')
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            elif isinstance(features, dict) and 'features' in features:
                features = features['features'].numpy()
            else:
                features = np.array(features)

        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        print(f"✅ Input data loaded: {features.shape}")
        return features

    except Exception as e:
        raise RuntimeError(f"Failed to load input data: {e}")


def predict_with_mlflow(model_uri: str, features: np.ndarray, batch_size: int = 32) -> dict:
    """
    Make predictions using MLflow model.

    Args:
        model_uri (str): MLflow model URI
        features (np.ndarray): Input features
        batch_size (int): Batch size for processing

    Returns:
        dict: Prediction results
    """
    try:
        print(f"🔄 Loading MLflow model from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"🚀 Starting batch prediction (batch_size={batch_size})")
        start_time = time.time()

        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []

        num_samples = len(features)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_features = features[i:batch_end]

            print(f"   Processing batch {i//batch_size + 1}/{num_batches} "
                  f"({batch_end-i} samples)")

            # Predict batch
            batch_start = time.time()
            batch_results = model.predict(None, batch_features)
            batch_time = time.time() - batch_start

            # Extract results
            if isinstance(batch_results, dict):
                all_predictions.extend(batch_results['predictions'])
                all_probabilities.extend(batch_results['probabilities'])
                all_confidence_scores.extend(batch_results['confidence_scores'])
            else:
                # Simple format fallback
                all_predictions.extend(batch_results)

        total_time = time.time() - start_time

        results = {
            'predictions': all_predictions,
            'probabilities': all_probabilities if all_probabilities else None,
            'confidence_scores': all_confidence_scores if all_confidence_scores else None,
            'metadata': {
                'model_uri': model_uri,
                'num_samples': num_samples,
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_sample': total_time / num_samples
            }
        }

        print(f"✅ Batch prediction completed")
        print(f"   Total samples: {num_samples}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per sample: {total_time/num_samples:.4f}s")

        return results

    except Exception as e:
        raise RuntimeError(f"MLflow prediction failed: {e}")


def predict_with_pytorch(model_path: str, features: np.ndarray, batch_size: int = 32) -> dict:
    """
    Make predictions using PyTorch model.

    Args:
        model_path (str): Path to model checkpoint
        features (np.ndarray): Input features
        batch_size (int): Batch size for processing

    Returns:
        dict: Prediction results
    """
    try:
        print(f"🔄 Loading PyTorch model from {model_path}")
        predictor = ContactPredictor(model_path=model_path)

        print(f"🚀 Starting batch prediction (batch_size={batch_size})")
        start_time = time.time()

        all_predictions = []
        all_probabilities = []
        all_confidence_scores = []

        num_samples = len(features)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_features = features[i:batch_end]

            print(f"   Processing batch {i//batch_size + 1}/{num_batches} "
                  f"({batch_end-i} samples)")

            # Predict batch
            batch_results = predictor._predict_batch(batch_features)

            # Extract results
            all_predictions.extend(batch_results['predictions'])
            all_probabilities.extend(batch_results['probabilities'])
            all_confidence_scores.extend(batch_results['confidence_scores'])

        total_time = time.time() - start_time

        results = {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'confidence_scores': all_confidence_scores,
            'metadata': {
                'model_path': model_path,
                'num_samples': num_samples,
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_sample': total_time / num_samples,
                'threshold': predictor.threshold,
                'device': str(predictor.device)
            }
        }

        print(f"✅ Batch prediction completed")
        print(f"   Total samples: {num_samples}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per sample: {total_time/num_samples:.4f}s")

        return results

    except Exception as e:
        raise RuntimeError(f"PyTorch prediction failed: {e}")


def save_results(results: dict, output_path: str):
    """
    Save prediction results to file.

    Args:
        results (dict): Prediction results
        output_path (str): Output file path
    """
    try:
        print(f"💾 Saving results to {output_path}")

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key != 'metadata':
                if isinstance(value, list):
                    # Handle list of numpy arrays
                    json_results[key] = [
                        arr.tolist() if hasattr(arr, 'tolist') else arr
                        for arr in value
                    ]
                elif hasattr(value, 'tolist'):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            else:
                json_results[key] = value

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"✅ Results saved to {output_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to save results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch prediction for ESM2 contact prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-uri', type=str,
                             help='MLflow model URI')
    model_group.add_argument('--model-path', type=str,
                             help='Path to PyTorch model checkpoint')

    # Data arguments
    parser.add_argument('--input-data', type=str, required=True,
                        help='Path to input data (.h5, .npy, .json, .pt)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save predictions (.json)')

    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for inference')

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.input_data).exists():
        print(f"❌ Input file not found: {args.input_data}")
        return 1

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load input data
        features = load_input_data(args.input_data)

        # Make predictions
        if args.model_uri:
            results = predict_with_mlflow(
                model_uri=args.model_uri,
                features=features,
                batch_size=args.batch_size
            )
        else:
            results = predict_with_pytorch(
                model_path=args.model_path,
                features=features,
                batch_size=args.batch_size
            )

        # Save results
        save_results(results, args.output)

        print(f"\n🎉 Batch prediction completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())