#!/usr/bin/env python3
"""
Model Serving Script for ESM2 Contact Prediction

This script provides utilities for serving trained ESM2 contact prediction models
using MLflow PyFunc with REST API and batch inference capabilities.

Key Features:
- REST API server with health checks
- Batch inference interface
- Model loading from MLflow registry
- Performance monitoring
- Input validation and error handling

Usage Examples:
    # Serve model via REST API
    uv run python scripts/serve_model.py \
        --model-uri "models:/esm2_contact/Production" \
        --port 5000

    # Batch inference
    uv run python scripts/serve_model.py \
        --model-uri "runs:/.../model" \
        --input-data data/test_features.h5 \
        --output-predictions predictions.json

    # Model validation
    uv run python scripts/serve_model.py \
        --model-uri "models:/esm2_contact/Production" \
        --validate-only
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import torch
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.esm2_contact.serving.contact_predictor import ContactPredictor
except ImportError:
    warnings.warn("Could not import ContactPredictor. Serving may be limited.")


class ContactPredictionServer:
    """
    REST API server for ESM2 contact prediction.

    Provides HTTP endpoints for model inference, health checks, and model information.
    """

    def __init__(self, model_uri: str, host: str = "127.0.0.1", port: int = 5000):
        """
        Initialize prediction server.

        Args:
            model_uri (str): MLflow model URI
            host (str): Server host
            port (int): Server port
        """
        self.model_uri = model_uri
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.model = None
        self.load_time = None
        self.request_count = 0
        self.start_time = time.time()

        # Load model
        self._load_model()

        # Setup routes
        self._setup_routes()

    def _load_model(self):
        """Load model from MLflow."""
        try:
            start_time = time.time()
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            self.load_time = time.time() - start_time

            print(f"✅ Model loaded successfully from {self.model_uri}")
            print(f"   Load time: {self.load_time:.2f}s")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model_uri': self.model_uri,
                'uptime': time.time() - self.start_time,
                'load_time': self.load_time,
                'request_count': self.request_count
            })

        @self.app.route('/info', methods=['GET'])
        def model_info():
            """Model information endpoint."""
            try:
                # Try to get model metadata
                metadata = {}
                if hasattr(self.model, 'metadata'):
                    metadata = self.model.metadata

                return jsonify({
                    'model_uri': self.model_uri,
                    'metadata': metadata,
                    'load_time': self.load_time,
                    'uptime': time.time() - self.start_time
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint."""
            try:
                # Validate request
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400

                data = request.get_json()

                # Validate input
                if 'features' not in data:
                    return jsonify({'error': 'Missing "features" field'}), 400

                features = data['features']

                # Convert to numpy if needed
                if isinstance(features, list):
                    features = np.array(features)

                # Make prediction
                start_time = time.time()
                results = self.model.predict(None, features)
                inference_time = time.time() - start_time

                self.request_count += 1

                return jsonify({
                    'predictions': results,
                    'inference_time': inference_time,
                    'request_id': self.request_count
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """Batch prediction endpoint for multiple proteins."""
            try:
                # Validate request
                if not request.is_json:
                    return jsonify({'error': 'Request must be JSON'}), 400

                data = request.get_json()

                # Validate input
                if 'batch_features' not in data:
                    return jsonify({'error': 'Missing "batch_features" field'}), 400

                batch_features = data['batch_features']

                if not isinstance(batch_features, list):
                    return jsonify({'error': 'batch_features must be a list'}), 400

                # Process batch
                start_time = time.time()
                results = []
                total_inference_time = 0

                for i, features in enumerate(batch_features):
                    # Convert to numpy if needed
                    if isinstance(features, list):
                        features = np.array(features)

                    # Make prediction
                    inference_start = time.time()
                    prediction = self.model.predict(None, features)
                    inference_time = time.time() - inference_start
                    total_inference_time += inference_time

                    results.append({
                        'index': i,
                        'prediction': prediction,
                        'inference_time': inference_time
                    })

                total_time = time.time() - start_time

                return jsonify({
                    'results': results,
                    'batch_size': len(batch_features),
                    'total_inference_time': total_inference_time,
                    'total_time': total_time,
                    'avg_inference_time': total_inference_time / len(batch_features)
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def run(self, debug: bool = False):
        """Run the server."""
        print(f"🚀 Starting contact prediction server")
        print(f"   Model: {self.model_uri}")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Debug: {debug}")

        self.app.run(host=self.host, port=self.port, debug=debug)


def batch_inference(model_uri: str,
                   input_path: str,
                   output_path: str,
                   batch_size: int = 32):
    """
    Perform batch inference on saved data.

    Args:
        model_uri (str): MLflow model URI
        input_path (str): Path to input data
        output_path (str): Path to save predictions
        batch_size (int): Batch size for inference
    """
    try:
        print(f"📊 Loading model from {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"📊 Loading input data from {input_path}")

        # Load input data (supporting different formats)
        if input_path.endswith('.h5'):
            import h5py
            with h5py.File(input_path, 'r') as f:
                if 'features' in f:
                    features = f['features'][:]
                else:
                    # Assume all datasets are features
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
        else:
            raise ValueError(f"Unsupported input format: {input_path}")

        print(f"✅ Input data loaded: {features.shape}")

        # Perform batch inference
        print(f"🚀 Starting batch inference (batch_size={batch_size})")
        start_time = time.time()

        all_results = []
        num_samples = len(features)

        for i in range(0, num_samples, batch_size):
            batch_end = min(i + batch_size, num_samples)
            batch_features = features[i:batch_end]

            print(f"   Processing batch {i//batch_size + 1}/{(num_samples-1)//batch_size + 1} "
                  f"({batch_end-i} samples)")

            # Predict batch
            batch_start = time.time()
            batch_results = model.predict(None, batch_features)
            batch_time = time.time() - batch_start

            # Store results with metadata
            for j, result in enumerate(batch_results):
                all_results.append({
                    'sample_index': i + j,
                    'predictions': result,
                    'batch_time': batch_time,
                    'batch_index': i // batch_size
                })

        total_time = time.time() - start_time

        print(f"✅ Batch inference completed")
        print(f"   Total samples: {num_samples}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per sample: {total_time/num_samples:.4f}s")

        # Save results
        print(f"💾 Saving results to {output_path}")
        with open(output_path, 'w') as f:
            json.dump({
                'model_uri': model_uri,
                'input_path': input_path,
                'output_path': output_path,
                'num_samples': num_samples,
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_sample': total_time / num_samples,
                'results': all_results
            }, f, indent=2)

        print(f"✅ Results saved to {output_path}")

    except Exception as e:
        raise RuntimeError(f"Batch inference failed: {e}")


def validate_model(model_uri: str):
    """
    Validate model loading and basic functionality.

    Args:
        model_uri (str): MLflow model URI
    """
    try:
        print(f"🔍 Validating model: {model_uri}")

        # Load model
        start_time = time.time()
        model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - start_time

        print(f"✅ Model loaded successfully")
        print(f"   Load time: {load_time:.2f}s")

        # Test with dummy input
        print("🧪 Testing with dummy input...")
        dummy_input = np.random.randn(1, 68, 64, 64).astype(np.float32)

        pred_start = time.time()
        try:
            prediction = model.predict(None, dummy_input)
            pred_time = time.time() - pred_start

            print(f"✅ Dummy prediction successful")
            print(f"   Prediction time: {pred_time:.4f}s")

            # Validate prediction format
            if isinstance(prediction, dict):
                required_keys = ['predictions', 'probabilities', 'confidence_scores']
                for key in required_keys:
                    if key not in prediction:
                        print(f"⚠️  Missing key in prediction: {key}")
                    else:
                        print(f"✅ Found key: {key}")

                print(f"   Prediction shape: {np.array(prediction['predictions']).shape}")
            else:
                print(f"⚠️  Unexpected prediction format: {type(prediction)}")

        except Exception as e:
            print(f"❌ Dummy prediction failed: {e}")

        print(f"✅ Model validation completed")

    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Model serving utilities for ESM2 contact prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start REST API server')
    serve_parser.add_argument('--model-uri', type=str, required=True,
                             help='MLflow model URI')
    serve_parser.add_argument('--host', type=str, default='127.0.0.1',
                             help='Server host')
    serve_parser.add_argument('--port', type=int, default=5000,
                             help='Server port')
    serve_parser.add_argument('--debug', action='store_true',
                             help='Enable debug mode')

    # Batch inference command
    batch_parser = subparsers.add_parser('batch', help='Batch inference')
    batch_parser.add_argument('--model-uri', type=str, required=True,
                              help='MLflow model URI')
    batch_parser.add_argument('--input-data', type=str, required=True,
                              help='Path to input data')
    batch_parser.add_argument('--output-predictions', type=str, required=True,
                              help='Path to save predictions')
    batch_parser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size for inference')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate model')
    validate_parser.add_argument('--model-uri', type=str, required=True,
                                 help='MLflow model URI')

    args = parser.parse_args()

    if args.command == 'serve':
        try:
            server = ContactPredictionServer(
                model_uri=args.model_uri,
                host=args.host,
                port=args.port
            )
            server.run(debug=args.debug)

        except Exception as e:
            print(f"❌ Server failed to start: {e}")
            return 1

    elif args.command == 'batch':
        try:
            batch_inference(
                model_uri=args.model_uri,
                input_path=args.input_data,
                output_path=args.output_predictions,
                batch_size=args.batch_size
            )

        except Exception as e:
            print(f"❌ Batch inference failed: {e}")
            return 1

    elif args.command == 'validate':
        try:
            validate_model(args.model_uri)

        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return 1

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())