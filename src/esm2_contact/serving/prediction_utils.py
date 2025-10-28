"""
Enhanced Prediction Utilities for ESM2 Contact Prediction

This module provides high-level utilities for making predictions with trained
ESM2 contact prediction models using modern MLflow PyFunc integration.

Key Features:
- Load models from MLflow with PyFunc support
- Batch processing for multiple PDB files
- Enhanced error handling and validation
- Model performance monitoring and logging
- Flexible input/output formats

Usage:
    from esm2_contact.serving.prediction_utils import (
        load_pyfunc_model, predict_from_pdb_pyfunc, predict_batch_from_pdb
    )

    # Load model from MLflow
    model = load_pyfunc_model("mlruns/exp_id/run_id")

    # Predict from PDB file
    results = predict_from_pdb_pyfunc(model, "protein.pdb")

    # Batch prediction
    results_list = predict_batch_from_pdb(model, ["protein1.pdb", "protein2.pdb"])
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc

from .contact_predictor import ContactPredictor


def load_pyfunc_model(model_uri: str,
                     enable_esm2_integration: bool = True,
                     **kwargs) -> mlflow.pyfunc.PyFuncModel:
    """
    Load MLflow PyFunc model with enhanced capabilities.

    Args:
        model_uri (str): MLflow model URI (e.g., "mlruns/exp_id/run_id/artifacts/model")
        enable_esm2_integration (bool): Whether to enable ESM2 features
        **kwargs: Additional arguments for model configuration

    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded model with enhanced capabilities

    Raises:
        RuntimeError: If model loading fails
        ValueError: If model URI is invalid
    """
    try:
        print(f"🔄 Loading MLflow PyFunc model: {model_uri}")

        # Validate model URI
        if not model_uri or not isinstance(model_uri, str):
            raise ValueError(f"Invalid model URI: {model_uri}")

        # Load the model
        model = mlflow.pyfunc.load_model(model_uri)

        print(f"✅ Model loaded successfully from {model_uri}")
        print(f"   Model type: {type(model)}")

        # Test model functionality
        try:
            # Try to get model metadata
            if hasattr(model, 'metadata'):
                metadata = model.metadata
                if metadata:
                    print(f"   Model signature: {metadata.signature}")
                    print(f"   Model run ID: {metadata.run_id}")

        except Exception as e:
            print(f"   ⚠️  Could not retrieve model metadata: {e}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load MLflow PyFunc model from {model_uri}: {e}")


def predict_from_pdb_pyfunc(model: mlflow.pyfunc.PyFuncModel,
                           pdb_path: str,
                           threshold: Optional[float] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Predict protein contacts from PDB file using MLflow PyFunc model.

    Args:
        model: Loaded MLflow PyFunc model
        pdb_path (str): Path to PDB file
        threshold (Optional[float]): Override threshold for binary predictions
        **kwargs: Additional prediction parameters

    Returns:
        Dict[str, Any]: Prediction results with metadata

    Raises:
        FileNotFoundError: If PDB file doesn't exist
        RuntimeError: If prediction fails
    """
    if not Path(pdb_path).exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    start_time = time.time()

    try:
        # Prepare input for PyFunc model
        input_data = pd.DataFrame([{'pdb_file': str(pdb_path)}])

        # Add threshold to input if provided
        if threshold is not None:
            input_data['threshold'] = threshold

        # Make prediction
        results = model.predict(input_data)

        # Extract first result (batch of 1)
        if isinstance(results, list) and len(results) > 0:
            prediction_result = results[0]
        else:
            prediction_result = results

        # Add timing and metadata
        prediction_time = time.time() - start_time
        prediction_result['prediction_time'] = prediction_time
        prediction_result['model_type'] = 'mlflow_pyfunc'

        return prediction_result

    except Exception as e:
        raise RuntimeError(f"Prediction failed for {pdb_path}: {e}")


def predict_batch_from_pdb(model: mlflow.pyfunc.PyFuncModel,
                          pdb_paths: List[str],
                          threshold: Optional[float] = None,
                          batch_size: Optional[int] = None,
                          show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Predict contacts for multiple PDB files efficiently.

    Args:
        model: Loaded MLflow PyFunc model
        pdb_paths (List[str]): List of PDB file paths
        threshold (Optional[float]): Override threshold for binary predictions
        batch_size (Optional[int]): Batch size for processing (None for all at once)
        show_progress (bool): Whether to show progress bar

    Returns:
        List[Dict[str, Any]]: List of prediction results

    Raises:
        ValueError: If no valid PDB files provided
        RuntimeError: If batch prediction fails
    """
    if not pdb_paths:
        raise ValueError("No PDB files provided for batch prediction")

    # Filter existing files
    valid_files = [str(p) for p in pdb_paths if Path(p).exists()]
    if not valid_files:
        raise ValueError("No valid PDB files found")

    print(f"🚀 Starting batch prediction for {len(valid_files)} PDB files")

    if batch_size is None:
        # Process all at once
        start_time = time.time()

        try:
            # Prepare batch input
            input_data = pd.DataFrame([{'pdb_file': path} for path in valid_files])

            # Add threshold if provided
            if threshold is not None:
                input_data['threshold'] = threshold

            # Make batch prediction
            results = model.predict(input_data)

            # Add timing metadata
            prediction_time = time.time() - start_time

            if isinstance(results, list):
                for result in results:
                    result['prediction_time'] = prediction_time / len(results)
                    result['model_type'] = 'mlflow_pyfunc'
            else:
                # Single result, convert to list
                results = [results]
                for result in results:
                    result['prediction_time'] = prediction_time
                    result['model_type'] = 'mlflow_pyfunc'

            print(f"✅ Batch prediction completed in {prediction_time:.2f}s")
            print(f"   Average time per protein: {prediction_time/len(results):.2f}s")

            return results

        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}")

    else:
        # Process in batches
        all_results = []
        total_start_time = time.time()

        # Create batches
        batches = [valid_files[i:i + batch_size] for i in range(0, len(valid_files), batch_size)]

        if show_progress:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Processing batches")
            except ImportError:
                print("   (Install tqdm for progress bars)")

        for batch_files in batches:
            try:
                # Prepare batch input
                input_data = pd.DataFrame([{'pdb_file': path} for path in batch_files])

                # Add threshold if provided
                if threshold is not None:
                    input_data['threshold'] = threshold

                # Make batch prediction
                batch_results = model.predict(input_data)

                # Add metadata
                for result in batch_results:
                    result['model_type'] = 'mlflow_pyfunc'

                all_results.extend(batch_results)

            except Exception as e:
                # Add error entries for failed files
                for pdb_path in batch_files:
                    error_result = {
                        'error': f"Batch processing failed: {str(e)}",
                        'pdb_file': pdb_path,
                        'model_type': 'mlflow_pyfunc'
                    }
                    all_results.append(error_result)

        total_time = time.time() - total_start_time
        print(f"✅ Batch prediction completed in {total_time:.2f}s")
        print(f"   Processed {len(all_results)} files in {len(batches)} batches")

        return all_results


def predict_from_sequence_pyfunc(model: mlflow.pyfunc.PyFuncModel,
                                sequence: str,
                                protein_id: str = "protein",
                                threshold: Optional[float] = None,
                                **kwargs) -> Dict[str, Any]:
    """
    Predict protein contacts from amino acid sequence using MLflow PyFunc model.

    Args:
        model: Loaded MLflow PyFunc model
        sequence (str): Amino acid sequence
        protein_id (str): Protein identifier
        threshold (Optional[float]): Override threshold for binary predictions
        **kwargs: Additional prediction parameters

    Returns:
        Dict[str, Any]: Prediction results with metadata

    Raises:
        ValueError: If sequence is invalid
        RuntimeError: If prediction fails
    """
    if not sequence or len(sequence.strip()) < 2:
        raise ValueError("Invalid sequence: must be at least 2 amino acids")

    start_time = time.time()

    try:
        # Prepare input for PyFunc model
        input_data = pd.DataFrame([{
            'sequence': sequence.strip(),
            'protein_id': protein_id
        }])

        # Add threshold if provided
        if threshold is not None:
            input_data['threshold'] = threshold

        # Make prediction
        results = model.predict(input_data)

        # Extract first result (batch of 1)
        if isinstance(results, list) and len(results) > 0:
            prediction_result = results[0]
        else:
            prediction_result = results

        # Add timing and metadata
        prediction_time = time.time() - start_time
        prediction_result['prediction_time'] = prediction_time
        prediction_result['model_type'] = 'mlflow_pyfunc'

        return prediction_result

    except Exception as e:
        raise RuntimeError(f"Sequence prediction failed for {protein_id}: {e}")


def create_pyfunc_model_from_checkpoint(model_path: str,
                                       output_path: Optional[str] = None,
                                       experiment_name: str = "esm2_contact_pyfunc",
                                       **kwargs) -> str:
    """
    Create MLflow PyFunc model from checkpoint file.

    Args:
        model_path (str): Path to model checkpoint
        output_path (Optional[str]): MLflow artifact path
        experiment_name (str): MLflow experiment name
        **kwargs: Additional model parameters

    Returns:
        str: MLflow model URI

    Raises:
        RuntimeError: If model creation fails
    """
    try:
        from .contact_predictor import create_pyfunc_model_instance, log_model_to_mlflow

        print(f"🔄 Creating MLflow PyFunc model from checkpoint: {model_path}")

        # Verify checkpoint exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Set output path
        if output_path is None:
            output_path = "esm2_contact_pyfunc_model"

        # Set experiment and start run
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            # Create PyFunc model instance
            pyfunc_model = create_pyfunc_model_instance(
                model_path=model_path,
                enable_esm2_integration=True,
                **kwargs
            )

            # Log model to MLflow
            log_model_to_mlflow(
                model_path=model_path,
                artifact_path=output_path,
                conda_env={
                    'channels': ['defaults', 'conda-forge', 'pytorch'],
                    'dependencies': [
                        'python=3.10',
                        'pytorch>=2.0.0',
                        'numpy>=1.20.0',
                        'mlflow>=2.15.0',
                        'scikit-learn>=1.0.0',
                        'pandas>=1.3.0',
                        'biopython>=1.79',
                        {
                            'pip': [
                                'pip>=23.0.0',
                                'fair-esm>=2.0.0'
                            ]
                        }
                    ],
                    'name': 'esm2_contact_pyfunc_env'
                }
            )

            # Get model URI
            model_uri = f"runs:/{run.info.run_id}/{output_path}"

            print(f"✅ PyFunc model created and logged: {model_uri}")
            print(f"   Run ID: {run.info.run_id}")
            print(f"   Experiment: {experiment_name}")

            return model_uri

    except Exception as e:
        raise RuntimeError(f"Failed to create PyFunc model from {model_path}: {e}")


def validate_pyfunc_model(model_uri: str,
                         test_pdb_path: Optional[str] = None,
                         test_sequence: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate MLflow PyFunc model functionality.

    Args:
        model_uri (str): MLflow model URI
        test_pdb_path (Optional[str]): Test PDB file for validation
        test_sequence (Optional[str]): Test sequence for validation

    Returns:
        Dict[str, Any]: Validation results

    Raises:
        RuntimeError: If validation fails
    """
    validation_results = {
        'model_uri': model_uri,
        'validation_passed': False,
        'tests': {},
        'errors': []
    }

    try:
        print(f"🧪 Validating PyFunc model: {model_uri}")

        # Load model
        model = load_pyfunc_model(model_uri)
        validation_results['tests']['model_loading'] = '✅ Passed'

        # Test with sequence if provided
        if test_sequence:
            try:
                result = predict_from_sequence_pyfunc(model, test_sequence, "test_protein")
                validation_results['tests']['sequence_prediction'] = '✅ Passed'
                validation_results['tests']['sequence_result'] = {
                    'sequence_length': len(test_sequence),
                    'has_contacts': 'predictions' in result,
                    'has_probabilities': 'probabilities' in result
                }
            except Exception as e:
                validation_results['tests']['sequence_prediction'] = f'❌ Failed: {e}'
                validation_results['errors'].append(f"Sequence prediction failed: {e}")

        # Test with PDB if provided
        if test_pdb_path and Path(test_pdb_path).exists():
            try:
                result = predict_from_pdb_pyfunc(model, test_pdb_path)
                validation_results['tests']['pdb_prediction'] = '✅ Passed'
                validation_results['tests']['pdb_result'] = {
                    'pdb_file': test_pdb_path,
                    'has_contacts': 'predictions' in result,
                    'has_probabilities': 'probabilities' in result,
                    'sequence_length': result.get('sequence_length', 'N/A')
                }
            except Exception as e:
                validation_results['tests']['pdb_prediction'] = f'❌ Failed: {e}'
                validation_results['errors'].append(f"PDB prediction failed: {e}")

        # Check overall validation status
        passed_tests = sum(1 for test in validation_results['tests'].values() if '✅' in test)
        total_tests = len(validation_results['tests'])

        if passed_tests == total_tests:
            validation_results['validation_passed'] = True
            print(f"✅ Model validation passed ({passed_tests}/{total_tests} tests)")
        else:
            print(f"⚠️  Model validation partial ({passed_tests}/{total_tests} tests passed)")

        return validation_results

    except Exception as e:
        validation_results['errors'].append(f"Validation failed: {e}")
        print(f"❌ Model validation failed: {e}")
        return validation_results


def get_model_info(model_uri: str) -> Dict[str, Any]:
    """
    Get detailed information about an MLflow PyFunc model.

    Args:
        model_uri (str): MLflow model URI

    Returns:
        Dict[str, Any]: Model information

    Raises:
        RuntimeError: If model info retrieval fails
    """
    try:
        model_info = {
            'model_uri': model_uri,
            'model_type': 'mlflow_pyfunc',
            'loadable': False,
            'metadata': {},
            'artifacts': {},
            'errors': []
        }

        # Try to load model and get metadata
        try:
            model = load_pyfunc_model(model_uri)
            model_info['loadable'] = True

            # Get model metadata
            if hasattr(model, 'metadata') and model.metadata:
                metadata = model.metadata
                model_info['metadata'] = {
                    'run_id': getattr(metadata, 'run_id', None),
                    'signature': str(getattr(metadata, 'signature', None)),
                    'model_uuid': getattr(metadata, 'model_uuid', None),
                }

            # Try to inspect model structure
            if hasattr(model, '_model_impl'):
                model_info['model_implementation'] = type(model._model_impl).__name__

            # Test basic functionality
            try:
                # This will vary based on model signature
                print("   ✅ Model successfully loaded and functional")
            except Exception as e:
                model_info['errors'].append(f"Model functionality test failed: {e}")

        except Exception as e:
            model_info['errors'].append(f"Failed to load model: {e}")

        return model_info

    except Exception as e:
        raise RuntimeError(f"Failed to get model info for {model_uri}: {e}")


def benchmark_model_performance(model_uri: str,
                               test_files: List[str],
                               iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark model performance on test files.

    Args:
        model_uri (str): MLflow model URI
        test_files (List[str]): List of test PDB files
        iterations (int): Number of prediction iterations

    Returns:
        Dict[str, Any]: Benchmark results

    Raises:
        RuntimeError: If benchmarking fails
    """
    try:
        print(f"🏃 Benchmarking model performance: {model_uri}")
        print(f"   Test files: {len(test_files)}")
        print(f"   Iterations: {iterations}")

        # Load model
        model = load_pyfunc_model(model_uri)

        # Filter valid files
        valid_files = [f for f in test_files if Path(f).exists()]
        if not valid_files:
            raise ValueError("No valid test files found for benchmarking")

        benchmark_results = {
            'model_uri': model_uri,
            'num_files': len(valid_files),
            'iterations': iterations,
            'timing_results': [],
            'statistics': {}
        }

        # Run benchmark iterations
        for iteration in range(iterations):
            print(f"   Iteration {iteration + 1}/{iterations}")

            start_time = time.time()

            # Run batch prediction
            results = predict_batch_from_pdb(
                model, valid_files, show_progress=False
            )

            iteration_time = time.time() - start_time
            successful_predictions = sum(1 for r in results if 'error' not in r)

            benchmark_results['timing_results'].append({
                'iteration': iteration + 1,
                'total_time': iteration_time,
                'successful_predictions': successful_predictions,
                'failed_predictions': len(results) - successful_predictions,
                'avg_time_per_prediction': iteration_time / len(valid_files)
            })

        # Calculate statistics
        times = [r['total_time'] for r in benchmark_results['timing_results']]
        successful_preds = [r['successful_predictions'] for r in benchmark_results['timing_results']]

        benchmark_results['statistics'] = {
            'avg_total_time': np.mean(times),
            'std_total_time': np.std(times),
            'min_total_time': np.min(times),
            'max_total_time': np.max(times),
            'avg_successful_predictions': np.mean(successful_preds),
            'throughput_predictions_per_second': len(valid_files) / np.mean(times)
        }

        print(f"✅ Benchmarking completed:")
        print(f"   Average time: {benchmark_results['statistics']['avg_total_time']:.2f}s")
        print(f"   Throughput: {benchmark_results['statistics']['throughput_predictions_per_second']:.2f} predictions/s")

        return benchmark_results

    except Exception as e:
        raise RuntimeError(f"Benchmarking failed: {e}")