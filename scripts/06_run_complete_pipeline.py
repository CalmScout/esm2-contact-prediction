#!/usr/bin/env python3
"""
Complete ESM2 Contact Prediction Pipeline

This script provides a unified pipeline that takes a directory of PDB files and
produces a trained model ready to serve via MLflow PyFunc. It integrates the
complete workflow from dataset generation through training and model logging.

Key Features:
- Single command from PDB files to MLflow-servable model
- Configurable dataset processing fraction (5% for quick testing)
- Complete MLflow integration with automatic model logging
- Memory-optimized processing with error handling
- Progress tracking and comprehensive logging

Pipeline Steps:
1. Setup configuration and MLflow experiment
2. Generate CNN dataset from PDB files
3. Train model with comprehensive tracking
4. Automatically log model to MLflow as PyFunc
5. Output model URI ready for serving

Usage Examples:
    # Quick test with 5% of data
    uv run python scripts/run_complete_pipeline.py \
        --pdb-dir data/train \
        --process-ratio 0.05 \
        --use-mlflow \
        --mlflow-experiment "protein_contact_prediction" \
        --epochs 10

    # Full pipeline with all data
    uv run python scripts/run_complete_pipeline.py \
        --pdb-dir data/train \
        --process-ratio 1.0 \
        --use-mlflow \
        --mlflow-experiment "protein_contact_prediction" \
        --epochs 50
"""

import os
import sys
import time
import argparse
import warnings
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
import mlflow
import mlflow.pyfunc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.esm2_contact.serving.contact_predictor import ContactPredictor, log_model_to_mlflow
except ImportError:
    warnings.warn("Could not import ContactPredictor. MLflow logging may be limited.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete ESM2 contact prediction pipeline from PDB to MLflow model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdb-dir data/train --process-ratio 0.05
  %(prog)s --pdb-dir data/train --process-ratio 0.1 --use-mlflow --mlflow-experiment "protein_contact_prediction" --epochs 20
        """
    )

    # Required arguments
    parser.add_argument(
        '--pdb-dir',
        type=str,
        required=True,
        help='Directory containing PDB files for dataset generation'
    )

    # Data processing arguments
    parser.add_argument(
        '--process-ratio',
        type=float,
        default=0.05,
        help='Fraction of PDB files to process (default: 0.05, i.e., 5%%)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible file selection (default: 42)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/pipeline_output',
        help='Output directory for generated dataset (default: data/pipeline_output)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default='esm2_contact_pipeline',
        help='Name for this experiment (used for outputs and MLflow)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Training epochs (default: 20)'
    )

    parser.add_argument(
        '--dataset-fraction',
        type=float,
        default=1.0,
        help='Fraction of generated dataset to use for training (default: 1.0)'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
    )

    # Model arguments
    parser.add_argument(
        '--base-channels',
        type=int,
        default=32,
        help='Base number of channels in CNN (default: 32)'
    )

    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.1,
        help='Dropout rate (default: 0.1)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )

    # MLflow arguments
    parser.add_argument(
        '--use-mlflow',
        action='store_true',
        help='Enable MLflow tracking and model logging'
    )

    parser.add_argument(
        '--mlflow-experiment',
        type=str,
        default='protein_contact_prediction',
        help='MLflow experiment name (default: protein_contact_prediction)'
    )

    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI (default: None)'
    )

    # Other arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--skip-dataset-generation',
        action='store_true',
        help='Skip dataset generation if dataset already exists'
    )

    return parser.parse_args()


def setup_mlflow(args: argparse.Namespace) -> bool:
    """Setup MLflow experiment and tracking."""
    if not args.use_mlflow:
        return False

    try:
        print(f"🔧 Setting up MLflow...")
        print(f"   Experiment: {args.mlflow_experiment}")
        if args.mlflow_tracking_uri:
            print(f"   Tracking URI: {args.mlflow_tracking_uri}")
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)

        mlflow.set_experiment(args.mlflow_experiment)
        print(f"   ✅ MLflow setup complete")
        return True

    except Exception as e:
        print(f"   ⚠️  MLflow setup failed: {e}")
        print(f"   Continuing without MLflow...")
        return False


def run_dataset_generation(args: argparse.Namespace) -> str:
    """Run dataset generation using the existing script."""
    print(f"\n📊 Step 1: Generating CNN Dataset from PDB Files")
    print(f"   PDB directory: {args.pdb_dir}")
    print(f"   Process ratio: {args.process_ratio * 100:.1f}%")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Output directory: {args.output_dir}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset file path
    dataset_path = output_path / "cnn_dataset.h5"

    # Check if dataset already exists
    if dataset_path.exists() and args.skip_dataset_generation:
        print(f"   ✅ Dataset already exists: {dataset_path}")
        print(f"   Skipping dataset generation as requested")
        return str(dataset_path)

    # Build command for dataset generation
    cmd = [
        sys.executable, str(project_root / "scripts" / "04_generate_cnn_dataset.py"),
        "--pdb-dir", str(args.pdb_dir),
        "--output-path", str(Path(args.output_dir) / "cnn_dataset.h5"),
        "--process-ratio", str(args.process_ratio),
        "--random-seed", str(args.random_seed)
    ]

    print(f"   🚀 Running: {' '.join(cmd)}")

    try:
        # Run dataset generation with real-time progress display
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            text=True,
            bufsize=1
        )

        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            output_lines.append(line.rstrip())
            # Always show progress output from dataset generation
            print(f"   {line.rstrip()}")

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        # Check if dataset was created
        if dataset_path.exists():
            print(f"   ✅ Dataset generated successfully: {dataset_path}")

            # Get dataset info
            import h5py
            try:
                with h5py.File(dataset_path, 'r') as f:
                    # Try different possible key names
                    if 'cnn_data' in f:
                        num_proteins = len(f['cnn_data'])
                    elif 'sequences' in f:
                        num_proteins = len(f['sequences'])
                    elif 'features' in f:
                        num_proteins = len(f['features'])
                    else:
                        num_proteins = "unknown"
                        print(f"   Available keys: {list(f.keys())}")
                    print(f"   📊 Dataset info: {num_proteins} proteins")
            except Exception as e:
                print(f"   ⚠️  Could not read dataset info: {e}")

            return str(dataset_path)
        else:
            raise RuntimeError("Dataset file was not created")

    except Exception as e:
        print(f"   ❌ Dataset generation failed:")
        print(f"   Error: {e}")
        raise RuntimeError(f"Dataset generation failed: {e}")


def run_training(args: argparse.Namespace, dataset_path: str, mlflow_enabled: bool) -> str:
    """Run model training using the existing script."""
    print(f"\n🏋️ Step 2: Training CNN Model")
    print(f"   Dataset: {dataset_path}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Dataset fraction: {args.dataset_fraction * 100:.1f}%")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Batch size: {args.batch_size}")

    # Build training command
    cmd = [
        sys.executable, str(project_root / "scripts" / "05_train_cnn.py"),
        "--dataset-path", dataset_path,
        "--experiment-name", args.experiment_name,
        "--epochs", str(args.epochs),
        "--dataset-fraction", str(args.dataset_fraction),
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
        "--base-channels", str(args.base_channels),
        "--dropout-rate", str(args.dropout_rate),
        "--learning-rate", str(args.learning_rate),
        "--batch-size", str(args.batch_size),
        "--random-seed", str(args.random_seed)
    ]

    if mlflow_enabled:
        cmd.extend([
            "--use-mlflow",
            "--mlflow-experiment", args.mlflow_experiment
        ])
        if args.mlflow_tracking_uri:
            cmd.append("--mlflow-tracking-uri")
            cmd.append(args.mlflow_tracking_uri)

    if args.verbose:
        cmd.append("--verbose")

    print(f"   🚀 Running: {' '.join(cmd)}")

    try:
        # Run training with real-time progress display
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            text=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            # Always show training progress output
            print(f"   {line.rstrip()}")

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        # Find the generated model file in experiment directory
        experiment_dir = Path("experiments") / args.experiment_name
        model_path = experiment_dir / "model.pth"

        if model_path.exists():
            print(f"   ✅ Model trained successfully: {model_path}")
            return str(model_path)
        else:
            print(f"   ⚠️  Model file not found at expected location: {model_path}")
            print(f"   Looking for any .pth files in experiment outputs...")
            all_pth_files = list(Path("experiments").glob(f"**/{args.experiment_name}*.pth"))
            if all_pth_files:
                model_path = all_pth_files[0]
                print(f"   ✅ Found model: {model_path}")
                return str(model_path)
            else:
                # Fallback: look for any recent .pth files
                recent_pth = list(Path("experiments").glob("**/model.pth"))
                if recent_pth:
                    model_path = recent_pth[-1]  # Most recent
                    print(f"   ✅ Found recent model: {model_path}")
                    return str(model_path)
                else:
                    raise RuntimeError("Could not find trained model file")

    except Exception as e:
        print(f"   ❌ Training failed:")
        print(f"   Error: {e}")
        raise RuntimeError(f"Training failed: {e}")


def log_model_to_mlflow_safely(model_path: str, args: argparse.Namespace) -> Optional[str]:
    """Log trained model to MLflow as PyFunc."""
    if not args.use_mlflow:
        print(f"\n📝 MLflow disabled - skipping model logging")
        return None

    print(f"\n📝 Step 3: Logging Model to MLflow as PyFunc")
    print(f"   Model: {model_path}")
    print(f"   Model name: {args.experiment_name}")

    try:
        # Start MLflow run for model logging
        with mlflow.start_run(run_name=f"{args.experiment_name}_pyfunc_logging") as run:
            print(f"   📊 MLflow run: {run.info.run_id}")

            # Log model parameters
            mlflow.log_params({
                "model_path": model_path,
                "experiment_name": args.experiment_name,
                "base_channels": args.base_channels,
                "dropout_rate": args.dropout_rate,
                "training_type": "complete_pipeline"
            })

            # Log model using ContactPredictor
            log_model_to_mlflow(
                model_path=model_path,
                artifact_path="contact_model",
                threshold=0.5,
                confidence_method="probability"
            )

            # Get model URI
            model_uri = f"runs:/{run.info.run_id}/contact_model"
            print(f"   ✅ Model logged to MLflow")
            print(f"   📍 Model URI: {model_uri}")

            return model_uri

    except Exception as e:
        print(f"   ❌ MLflow logging failed: {e}")
        print(f"   ⚠️  Model is still available as: {model_path}")
        return None


def print_final_results(args: argparse.Namespace, dataset_path: str, model_path: str, model_uri: Optional[str]):
    """Print final results and next steps."""
    print(f"\n🎉 Complete Pipeline Finished Successfully!")
    print(f"="*60)

    print(f"\n📊 Generated Resources:")
    print(f"   🗂️  Dataset: {dataset_path}")
    print(f"   🧠 Model: {model_path}")

    if model_uri:
        print(f"   📍 MLflow Model URI: {model_uri}")

    print(f"\n⏱️  Pipeline Configuration:")
    print(f"   📁 PDB Directory: {args.pdb_dir}")
    print(f"   📈 Process Ratio: {args.process_ratio * 100:.1f}%")
    print(f"   🏋️  Training Epochs: {args.epochs}")
    print(f"   🎲 Dataset Fraction: {args.dataset_fraction * 100:.1f}%")
    print(f"   🧪 MLflow Enabled: {args.use_mlflow}")

    print(f"\n🚀 Next Steps:")

    if model_uri:
        print(f"\n1. Start REST API Server:")
        print(f"   uv run python scripts/serve_model.py serve \\")
        print(f"       --model-uri \"{model_uri}\" \\")
        print(f"       --port 5000")

        print(f"\n2. Make Predictions via API:")
        print(f"   curl -X POST http://localhost:5000/predict \\")
        print(f"        -H \"Content-Type: application/json\" \\")
        print(f"        -d '{{\"features\": [[...68-channel tensor...]]}}'")

        print(f"\n3. Batch Inference:")
        print(f"   uv run python scripts/serve_model.py batch \\")
        print(f"       --model-uri \"{model_uri}\" \\")
        print(f"       --input-data {dataset_path} \\")
        print(f"       --output-predictions predictions.json")

    else:
        print(f"\n1. Model Available for Manual Loading:")
        print(f"   Model file: {model_path}")
        print(f"   To enable MLflow serving, run with --use-mlflow")

    print(f"\n4. Model Validation:")
    print(f"   uv run python scripts/serve_model.py validate \\")
    if model_uri:
        print(f"       --model-uri \"{model_uri}\"")
    else:
        print(f"       --model-uri {model_path}")

    print(f"\n📚 For more usage examples, see README.md")


def main():
    """Main pipeline execution."""
    print(f"🚀 ESM2 Contact Prediction - Complete Pipeline")
    print(f"   From PDB files to MLflow-servable model")
    print(f"="*60)

    # Parse arguments
    args = parse_arguments()

    if args.verbose:
        print(f"\n📋 Pipeline Configuration:")
        for arg, value in vars(args).items():
            if arg != 'verbose':
                print(f"   {arg:20s}: {value}")

    # Validate arguments
    if not 0.0 < args.process_ratio <= 1.0:
        raise ValueError("process-ratio must be in range (0.0, 1.0]")

    if not 0.0 < args.dataset_fraction <= 1.0:
        raise ValueError("dataset-fraction must be in range (0.0, 1.0]")

    if not Path(args.pdb_dir).exists():
        raise ValueError(f"PDB directory does not exist: {args.pdb_dir}")

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    try:
        # Setup MLflow
        mlflow_enabled = setup_mlflow(args)

        # Step 1: Generate dataset
        dataset_path = run_dataset_generation(args)

        # Step 2: Train model
        model_path = run_training(args, dataset_path, mlflow_enabled)

        # Step 3: Log to MLflow
        model_uri = log_model_to_mlflow_safely(model_path, args)

        # Print final results
        print_final_results(args, dataset_path, model_path, model_uri)

        print(f"\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            print(f"\nFull traceback:")
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())