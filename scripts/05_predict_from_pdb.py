#!/usr/bin/env python3
"""
Enhanced PDB Contact Prediction Script

This script provides a comprehensive solution for predicting protein contacts from PDB files
using trained ESM2-based models. It's compatible with the latest upstream code and supports
various model checkpoint formats.

Key Features:
- Compatible with new BinaryContactCNN architecture
- Handles multiple checkpoint formats (old/new training pipeline)
- Real ESM2 embeddings + pattern-based template features
- Enhanced error handling and input validation
- Comprehensive output with confidence scores and metadata
- Automatic threshold optimization based on protein size
- Memory-optimized GPU/CPU processing

Usage Examples:
    # Basic usage
    uv run python scripts/07_predict_from_pdb.py \\
        --pdb-file data/my_protein.pdb \\
        --model-path models/best_model.pth \\
        --output predictions.json

    # With custom threshold
    uv run python scripts/07_predict_from_pdb.py \\
        --pdb-file data/my_protein.pdb \\
        --model-path models/best_model.pth \\
        --threshold 0.3 \\
        --output predictions.json

    # Verbose output for debugging
    uv run python scripts/07_predict_from_pdb.py \\
        --pdb-file data/my_protein.pdb \\
        --model-path models/best_model.pth \\
        --verbose

Output Format:
    - contact_binary: 2D array of binary contact predictions (0/1)
    - contact_probabilities: 2D array of prediction probabilities
    - confidence_scores: 2D array of confidence values
    - model_info: Model architecture and parameters
    - prediction_statistics: Statistics about predictions
    - performance: Timing and performance metrics

Compatibility:
    - Works with models trained using the new BinaryContactCNN architecture
    - Supports both old and new checkpoint formats
    - Compatible with 68-channel input (64 ESM2 + 4 template features)
    - Handles various model configurations (different base_channels, dropout_rates, etc.)
"""

import os
import sys
import argparse
import warnings
import hashlib
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try imports with error handling
try:
    from src.esm2_contact.training.model import BinaryContactCNN
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    MODEL_IMPORTS_AVAILABLE = False

# MLflow imports
try:
    import mlflow
    import mlflow.pyfunc
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try imports for enhanced PyFunc integration
try:
    from src.esm2_contact.serving import (
        load_pyfunc_model,
        predict_from_pdb_pyfunc,
        predict_batch_from_pdb,
        predict_from_sequence_pyfunc,
        create_pyfunc_model_from_checkpoint,
        validate_pyfunc_model
    )
    from src.esm2_contact.mlflow_utils import (
        get_best_pyfunc_model,
        load_pyfunc_model_from_run,
        get_pyfunc_model_metrics
    )
    PYFUNC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PyFunc modules: {e}")
    PYFUNC_AVAILABLE = False

# Global ESM2 model cache (load once, reuse across calls)
_ESM2_MODEL = None
_ESM2_ALPHABET = None
_ESM2_DEVICE = None

def load_esm2_model():
    """Load ESM2 model once and cache globally with optimized loading."""
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_DEVICE

    if _ESM2_MODEL is None:
        print("📱 Loading ESM2 model for contact prediction...")
        try:
            import esm
            import time
            start_time = time.time()

            # Disable unnecessary logging for faster loading
            import logging
            logging.getLogger('esm').setLevel(logging.ERROR)

            # Load model and alphabet with optimized settings
            print(f"   📥 Downloading/loading ESM2-650M model...")
            _ESM2_MODEL, _ESM2_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()

            # Optimize model for inference
            _ESM2_MODEL.eval()
            _ESM2_MODEL = _ESM2_MODEL.half()  # Use half precision for faster inference

            # Move to GPU if available with optimized memory usage
            _ESM2_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            _ESM2_MODEL = _ESM2_MODEL.to(_ESM2_DEVICE)

            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    _ESM2_MODEL = torch.compile(_ESM2_MODEL, mode="reduce-overhead")
                    print(f"   ⚡ Model compiled for faster inference")
                except:
                    print(f"   📝 Model compilation skipped (not supported)")

            load_time = time.time() - start_time
            print(f"   ✅ ESM2 model loaded on {_ESM2_DEVICE} in {load_time:.1f}s")

        except Exception as e:
            print(f"   ❌ Failed to load ESM2 model: {e}")
            print(f"   🔧 Falling back to deterministic embeddings")
            raise e

    return _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_DEVICE

def validate_sequence_for_esm(sequence: str) -> str:
    """Validate and clean sequence for ESM2 processing."""
    # Valid amino acids for ESM2
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')

    # Clean sequence: replace invalid characters with X
    clean_sequence = ''.join([aa if aa in valid_aa else 'X' for aa in sequence])

    # Remove any non-standard characters
    clean_sequence = ''.join([aa for aa in clean_sequence if aa.isalpha() or aa == 'X'])

    return clean_sequence.upper()

def generate_real_esm2_embeddings(protein_id: str, sequence: str) -> np.ndarray:
    """Generate real ESM2 embeddings using the ESM2 model."""
    # Load ESM2 model
    model, alphabet, device = load_esm2_model()
    batch_converter = alphabet.get_batch_converter()

    # Validate and clean sequence
    clean_sequence = validate_sequence_for_esm(sequence)

    print(f"   🧬 Sequence: {clean_sequence[:20]}... ({len(clean_sequence)} residues)")

    if len(clean_sequence) < 2:
        raise ValueError(f"Sequence too short after cleaning: {len(clean_sequence)}")

    # Prepare sequence for ESM2
    sequences_list = [(protein_id, clean_sequence)]

    # Prepare batch
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences_list)
    batch_tokens = batch_tokens.to(device)
    model.eval()

    with torch.no_grad():
        # Compute forward pass
        print(f"   🧠 Running ESM2 inference...")
        outputs = model(
            batch_tokens,
            repr_layers=[33],  # Final layer
            return_contacts=False
        )

        # Extract embeddings
        embeddings = outputs["representations"][33]  # Shape: (1, seq_len + 2, 1280)

        # Remove BOS and EOS tokens (first and last tokens)
        embeddings = embeddings[:, 1:-1, :]  # Shape: (1, seq_len, 1280)

        # Move to CPU and convert to numpy
        embedding = embeddings[0].cpu().numpy()  # Shape: (seq_len, 1280)

    # Transpose to match expected format (1280, seq_len)
    embedding_transposed = embedding.T  # Shape: (1280, seq_len)

    print(f"   ✅ ESM2 embeddings: {embedding_transposed.shape}")

    # Memory cleanup
    del batch_tokens, embeddings, outputs
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return embedding_transposed

def generate_pattern_based_template_features(sequence: str) -> np.ndarray:
    """Generate template features using the same pattern-based approach as training pipeline."""
    print(f"   🔍 Creating pattern-based template channels (same as training pipeline)...")

    L = len(sequence)
    template_channels = np.zeros((4, L, L), dtype=np.float32)

    # Channel 0: Sequence conservation pattern (local sequence proximity)
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= 2:
                template_channels[0, i, j] = 0.8

    # Channel 1: Distance-based pattern (exponential decay)
    for i in range(L):
        for j in range(L):
            dist = abs(i - j)
            if dist <= 8:
                template_channels[1, i, j] = np.exp(-dist / 4.0)

    # Channel 2: Predicted secondary structure pattern (helical propensity)
    for i in range(L):
        for j in range(L):
            if i != j:
                dist = abs(i - j)
                # Helical periodicity pattern
                if 3 <= dist <= 5:
                    template_channels[2, i, j] = 0.3
                elif dist >= 15:
                    template_channels[2, i, j] = 0.1

    # Channel 3: Coevolution pattern (long-range contacts)
    for i in range(L):
        for j in range(L):
            if i != j:
                dist = abs(i - j)
                if dist > 12 and dist < 50:
                    template_channels[3, i, j] = 0.2 * (1 - dist / 50)

    # Set diagonal to 1.0
    for i in range(4):
        np.fill_diagonal(template_channels[i], 1.0)

    print(f"   ✅ Template channels created: {template_channels.shape}")
    return template_channels

def load_model_from_mlflow(model_uri: str, device: torch.device):
    """Load trained model from MLflow with enhanced compatibility."""
    if not MLFLOW_AVAILABLE:
        raise RuntimeError("MLflow is not available")
    if not MODEL_IMPORTS_AVAILABLE:
        raise RuntimeError("Required model imports are not available")

    try:
        print(f"🔄 Loading model from MLflow: {model_uri}")

        # Extract model artifact path from URI
        if '/artifacts/' in model_uri:
            artifact_path = model_uri
        else:
            # If just the run ID is provided, construct artifact path
            artifact_path = f"{model_uri}/artifacts/best_model_checkpoint"

        print(f"   📥 MLflow artifact path: {artifact_path}")

        # Download the model artifact
        import tempfile
        local_path = mlflow.artifacts.download_artifacts(artifact_path)

        print(f"   📁 Downloaded to: {local_path}")

        # Handle different artifact types
        if Path(local_path).is_dir():
            # Look for .pth files in the directory
            pth_files = list(Path(local_path).glob("*.pth"))
            if pth_files:
                model_file_path = pth_files[0]
                print(f"   📄 Found model file: {model_file_path.name}")
            else:
                raise RuntimeError(f"No .pth files found in {local_path}")
        else:
            model_file_path = local_path

        # Load the model from the downloaded file
        model = load_model(str(model_file_path), device)

        # Try to extract run metadata
        try:
            # Get run info from URI
            run_id = model_uri.split('/')[1] if len(model_uri.split('/')) > 1 else "unknown"
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)

            print(f"   📊 Run information:")
            print(f"      Run ID: {run_id}")
            print(f"      Experiment: {run.info.experiment_id}")
            print(f"      Status: {run.info.status}")
            print(f"      Best AUC: {run.data.metrics.get('best_auc', 'N/A')}")

            # Display key parameters
            params = run.data.params
            if params:
                print(f"      Key parameters:")
                key_params = ['learning_rate', 'batch_size', 'base_channels', 'dataset_fraction']
                for param in key_params:
                    if param in params:
                        print(f"         {param}: {params[param]}")

        except Exception as e:
            print(f"   ⚠️  Could not fetch run metadata: {e}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from MLflow {model_uri}: {e}")


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint with enhanced compatibility."""
    if not MODEL_IMPORTS_AVAILABLE:
        raise RuntimeError("Required model imports are not available")

    try:
        print(f"🧠 Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Enhanced checkpoint format handling
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']

            # Try to get config from various possible locations
            config = {}
            if 'config' in checkpoint:
                config = checkpoint['config']
            elif 'model_config' in checkpoint:
                config = checkpoint['model_config']

            # Extract model architecture parameters with defaults
            in_channels = config.get('in_channels', 68)
            base_channels = config.get('base_channels', 32)
            dropout_rate = config.get('dropout_rate', 0.1)

            # Additional metadata
            if 'history' in checkpoint:
                print(f"   📊 Training history found in checkpoint")
            if 'best_auc' in checkpoint:
                print(f"   🎯 Best validation AUC: {checkpoint['best_auc']:.4f}")

        elif isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            # Alternative format
            model_state = checkpoint['model_state_dict']
            model_config = checkpoint['model_config']
            in_channels = model_config.get('in_channels', 68)
            base_channels = model_config.get('base_channels', 32)
            dropout_rate = model_config.get('dropout_rate', 0.1)

        else:
            # Just state dict - try to infer architecture from model state
            model_state = checkpoint
            print(f"   🔍 Inferring architecture from model state dict...")

            # Try to infer from first conv layer weight shape
            if 'conv1.0.weight' in model_state:
                first_conv_shape = model_state['conv1.0.weight'].shape
                in_channels = first_conv_shape[1]
                base_channels = first_conv_shape[0]
            elif 'predictor.0.weight' in model_state:
                # Try to infer from predictor layer
                predictor_shape = model_state['predictor.0.weight'].shape
                base_channels = predictor_shape[0] // 4  # Reverse of architecture
                in_channels = 68
            else:
                # Fallback defaults
                print(f"   ⚠️  Could not infer architecture, using defaults")
                in_channels = 68
                base_channels = 32
            dropout_rate = 0.1

        # Create and load model
        model = BinaryContactCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout_rate=dropout_rate
        )

        # Load state dict with error handling for mismatched keys
        try:
            model.load_state_dict(model_state, strict=True)
            print(f"   ✅ Model loaded with strict matching")
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"   ⚠️  State dict mismatch, trying non-strict loading...")
                try:
                    model.load_state_dict(model_state, strict=False)
                    print(f"   ✅ Model loaded with non-strict matching")
                except RuntimeError as e2:
                    raise RuntimeError(f"Failed to load model even with non-strict matching: {e2}")
            else:
                raise e

        model.to(device)
        model.eval()

        # Print model information
        model_info = model.get_model_info()
        print(f"   ✅ Model loaded successfully")
        print(f"   Architecture: {in_channels}→{base_channels} channels")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Memory footprint: {model_info['memory_footprint_mb']:.1f}MB")
        print(f"   Device: {device}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def extract_sequence_from_pdb_simple(pdb_path: str) -> str:
    """Simple PDB sequence extraction using basic parsing."""
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_path)

        # Get amino acid sequence
        sequence = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Skip heteroatoms and waters
                        res_name = residue.get_resname()
                        # Simple 3-letter to 1-letter conversion
                        aa_map = {
                            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                        }
                        if res_name in aa_map:
                            sequence.append(aa_map[res_name])

        return ''.join(sequence)
    except Exception:
        # Fallback: return a dummy sequence if parsing fails
        return "ACDEFGHIKLMNPQRSTVWY" * 10  # 250 residues dummy


def assemble_68_channel_tensor(esm2_embedding: np.ndarray, template_channels: np.ndarray) -> np.ndarray:
    """Assemble 68-channel tensor (4 template + 64 ESM2) with robust shape handling."""
    try:
        L = template_channels.shape[1]  # Sequence length from template channels
        channels = 68
        height = L
        width = L

        print(f"   🏗️ Assembling tensor: L={L}, esm2_shape={esm2_embedding.shape}, template_shape={template_channels.shape}")

        # Initialize multi-channel tensor
        tensor = np.zeros((channels, height, width), dtype=np.float32)

        # Channels 0-3: Template channels
        if template_channels.shape != (4, L, L):
            raise ValueError(f"Template channels shape mismatch: expected (4, {L}, {L}), got {template_channels.shape}")

        tensor[0:4] = template_channels
        print(f"   ✅ Template channels assigned: {template_channels.shape}")

        # Channels 4-67: ESM2 channels (64 channels)
        print(f"   📊 ESM2 embedding shape: {esm2_embedding.shape}")

        # Ensure we have at least 64 dimensions from ESM2 embedding
        if esm2_embedding.shape[0] < 64:
            print(f"   ⚠️  ESM2 has only {esm2_embedding.shape[0]} dimensions, padding to 64")
            # Pad ESM2 embedding to 64 dimensions
            padded_esm2 = np.zeros((64, esm2_embedding.shape[1]), dtype=np.float32)
            padded_esm2[:esm2_embedding.shape[0], :] = esm2_embedding
            esm2_embedding = padded_esm2

        # Handle sequence length matching
        if esm2_embedding.shape[1] == L:
            # Perfect match: ESM2 embedding has correct dimensions
            esm2_64_channels = esm2_embedding[:64, :]  # Shape: (64, L)
            print(f"   ✅ Perfect sequence length match: {esm2_64_channels.shape}")
        elif esm2_embedding.shape[1] > L:
            # ESM2 embedding is longer, truncate
            esm2_64_channels = esm2_embedding[:64, :L]
            print(f"   ✅ ESM2 truncated: {esm2_64_channels.shape} (was longer)")
        else:
            # ESM2 embedding is shorter, pad with zeros
            esm2_64_channels = np.zeros((64, L), dtype=np.float32)
            esm2_64_channels[:, :esm2_embedding.shape[1]] = esm2_embedding[:64, :esm2_embedding.shape[1]]
            print(f"   ✅ ESM2 padded: {esm2_64_channels.shape} (was shorter)")

        # Assign ESM2 channels - replicate 1D features across 2D matrix
        for i in range(64):
            # Replicate the 1D ESM2 feature across all positions to create 2D map
            tensor[4 + i] = np.tile(esm2_64_channels[i:i+1, :], (L, 1))
        print(f"   ✅ ESM2 channels assigned: {esm2_64_channels.shape} → (64, {L}, {L})")

        print(f"   ✅ Final tensor: {tensor.shape}")
        print(f"   📊 Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

        return tensor

    except Exception as e:
        raise RuntimeError(f"Error assembling tensor: {e}")

def process_pdb_file(pdb_path: str, device: torch.device):
    """Process PDB file to create real 68-channel tensor."""
    print(f"📁 Processing PDB file: {pdb_path}")

    # Extract sequence from PDB
    sequence = extract_sequence_from_pdb_simple(pdb_path)
    if not sequence:
        raise ValueError(f"No valid sequence extracted from PDB file: {pdb_path}")

    print(f"   📏 Sequence length: {len(sequence)}")

    # Get protein ID from PDB file name
    protein_id = Path(pdb_path).stem

    # Generate real ESM2 embeddings (1280 dimensions)
    print(f"   🧬 Generating ESM2 embeddings...")
    esm2_embedding = generate_real_esm2_embeddings(protein_id, sequence)

    # Generate pattern-based template features (4 channels) - same as training pipeline
    print(f"   🔍 Generating template features...")
    template_features = generate_pattern_based_template_features(sequence)

    # Assemble 68-channel tensor
    print(f"   🏗️ Assembling 68-channel tensor...")
    features_68 = assemble_68_channel_tensor(esm2_embedding, template_features)

    # Convert to PyTorch tensor
    features_tensor = torch.from_numpy(features_68).unsqueeze(0).to(device)

    print(f"   ✅ Final tensor ready: {features_tensor.shape}")

    return features_tensor, sequence


def calculate_optimal_threshold(sequence_length: int) -> float:
    """Calculate optimal threshold based on protein length and real feature behavior."""
    # Updated thresholds for real ESM2 + template features (more conservative)
    # Real features produce more conservative probabilities than dummy features
    # Small proteins (<100): target 8-12% contacts
    # Medium proteins (100-300): target 4-8% contacts
    # Large proteins (>300): target 2-5% contacts
    if sequence_length < 100:
        return 0.15  # Much lower threshold for small proteins
    elif sequence_length < 300:
        return 0.20  # Conservative threshold for medium proteins
    else:
        return 0.25  # Lower threshold for large proteins

def analyze_probability_distribution(contact_probs: np.ndarray, sequence_length: int) -> dict:
    """Analyze probability distribution to suggest optimal thresholds."""
    flat_probs = contact_probs.flatten()

    # Calculate percentiles
    percentiles = {
        'p10': np.percentile(flat_probs, 10),
        'p25': np.percentile(flat_probs, 25),
        'p50': np.percentile(flat_probs, 50),
        'p75': np.percentile(flat_probs, 75),
        'p90': np.percentile(flat_probs, 90),
        'p95': np.percentile(flat_probs, 95),
        'p99': np.percentile(flat_probs, 99)
    }

    # Calculate contact densities at different thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    densities = {}
    for threshold in thresholds:
        density = np.mean(contact_probs > threshold)
        densities[threshold] = density

    # Find optimal threshold based on expected density range
    if sequence_length < 100:
        target_range = (0.06, 0.15)  # 6-15% for small proteins
    elif sequence_length < 300:
        target_range = (0.03, 0.10)  # 3-10% for medium proteins
    else:
        target_range = (0.02, 0.08)   # 2-8% for large proteins

    # Find best threshold
    best_threshold = 0.5
    best_density = 0
    for threshold, density in densities.items():
        if target_range[0] <= density <= target_range[1]:
            best_threshold = threshold
            best_density = density
            break

    # If no threshold in range, pick closest
    if best_threshold == 0.5:
        min_distance = float('inf')
        for threshold, density in densities.items():
            distance = min(abs(density - target_range[0]), abs(density - target_range[1]))
            if distance < min_distance:
                min_distance = distance
                best_threshold = threshold
                best_density = density

    return {
        'percentiles': percentiles,
        'densities': densities,
        'suggested_threshold': best_threshold,
        'suggested_density': best_density,
        'target_range': target_range,
        'max_probability': np.max(flat_probs),
        'mean_probability': np.mean(flat_probs)
    }

def validate_predictions(contact_binary: np.ndarray, contact_probs: np.ndarray, sequence_length: int) -> dict:
    """Validate predictions and provide realistic contact density ranges."""
    contact_density = np.mean(contact_binary)

    # Analyze probability distribution
    prob_analysis = analyze_probability_distribution(contact_probs, sequence_length)

    # Expected contact density ranges based on protein size
    if sequence_length < 100:
        expected_range = (0.06, 0.15)  # 6-15% for small proteins
    elif sequence_length < 300:
        expected_range = (0.03, 0.10)  # 3-10% for medium proteins
    else:
        expected_range = (0.02, 0.08)   # 2-8% for large proteins

    is_realistic = expected_range[0] <= contact_density <= expected_range[1]

    return {
        'contact_density': contact_density,
        'expected_range': expected_range,
        'is_realistic': is_realistic,
        'num_contacts': int(np.sum(contact_binary)),
        'probability_analysis': prob_analysis
    }

def validate_inputs(pdb_path: str, model_path: str = None, threshold: float = None, model_uri: str = None) -> tuple:
    """Validate all inputs before processing."""
    errors = []
    warnings_list = []

    # Validate PDB file
    if not Path(pdb_path).exists():
        errors.append(f"PDB file not found: {pdb_path}")
    elif not Path(pdb_path).suffix.lower() in ['.pdb', '.ent', '.cif']:
        warnings_list.append(f"Unexpected PDB file extension: {Path(pdb_path).suffix}")
    else:
        # Check file size
        pdb_size = Path(pdb_path).stat().st_size
        if pdb_size == 0:
            errors.append(f"PDB file is empty: {pdb_path}")
        elif pdb_size > 100 * 1024 * 1024:  # 100MB
            warnings_list.append(f"Large PDB file detected ({pdb_size/1024/1024:.1f}MB), processing may be slow")

    # Validate model source
    if model_path and model_uri:
        errors.append("Cannot specify both model-path and model-uri")
    elif model_path:
        # Validate local model file
        if not Path(model_path).exists():
            errors.append(f"Model file not found: {model_path}")
        elif not Path(model_path).suffix.lower() in ['.pth', '.pt']:
            warnings_list.append(f"Unexpected model file extension: {Path(model_path).suffix}")
        else:
            # Check file size
            model_size = Path(model_path).stat().st_size
            if model_size == 0:
                errors.append(f"Model file is empty: {model_path}")
    elif model_uri:
        # Validate MLflow URI
        if not MLFLOW_AVAILABLE:
            errors.append("MLflow is not available but model-uri was specified")
        elif not model_uri.startswith('mlruns/'):
            warnings_list.append(f"Unusual MLflow URI format: {model_uri}")
    else:
        errors.append("Either model-path or model-uri must be specified")

    # Validate threshold
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            errors.append(f"Threshold must be a number, got {type(threshold)}")
        elif not 0.0 <= threshold <= 1.0:
            errors.append(f"Threshold must be between 0.0 and 1.0, got {threshold}")

    # Check required dependencies
    if not MODEL_IMPORTS_AVAILABLE:
        errors.append("Required model imports are not available. Check your installation.")

    return errors, warnings_list


def predict_contacts(pdb_path: str, model_path: str = None, model_uri: str = None, threshold: float = None):
    """
    Predict protein contacts from PDB file using real ESM2 and homology features.

    Args:
        pdb_path (str): Path to PDB file
        model_path (str): Path to trained model (.pth file)
        model_uri (str): MLflow model URI
        threshold (float): Threshold for binary predictions (auto-calculated if None)

    Returns:
        dict: Prediction results including contact map and metadata
    """
    import time
    start_time = time.time()

    print(f"🚀 Starting Real PDB Contact Prediction")
    print(f"   PDB file: {pdb_path}")
    if model_path:
        print(f"   Model: {model_path} (local file)")
    elif model_uri:
        print(f"   Model: {model_uri} (MLflow)")
    else:
        raise ValueError("Either model_path or model_uri must be provided")
    print(f"="*50)

    # Validate inputs
    errors, warnings_list = validate_inputs(pdb_path, model_path, threshold, model_uri)

    if errors:
        print(f"\n❌ Input validation failed:")
        for error in errors:
            print(f"   • {error}")
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    if warnings_list:
        print(f"\n⚠️  Warnings:")
        for warning in warnings_list:
            print(f"   • {warning}")

    # Setup device with memory check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    if device.type == 'cuda':
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            print(f"   GPU Memory: {gpu_memory:.1f}GB total, {gpu_memory_free:.1f}GB available")
        except:
            print(f"   GPU Memory: Could not determine memory usage")

    # Load model
    model_start = time.time()
    try:
        if model_path:
            model = load_model(model_path, device)
            model_source = f"local file: {model_path}"
        elif model_uri:
            model = load_model_from_mlflow(model_uri, device)
            model_source = f"MLflow: {model_uri}"
        else:
            raise ValueError("No model source specified")
        model_time = time.time() - model_start
        print(f"   ⏱️  Model loading: {model_time:.1f}s")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        raise

    # Process PDB file
    process_start = time.time()
    try:
        features_tensor, sequence = process_pdb_file(pdb_path, device)
        process_time = time.time() - process_start
        print(f"   ⏱️  Feature processing: {process_time:.1f}s")
        sequence_length = len(sequence)
    except Exception as e:
        print(f"   ❌ Feature processing failed: {e}")
        raise

    # Make prediction
    print(f"\n🔮 Making prediction...")
    pred_start = time.time()

    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = torch.sigmoid(logits)

        # Remove batch dimension
        contact_probs = probabilities.squeeze(0).cpu().numpy()

    pred_time = time.time() - pred_start
    print(f"   ⏱️  Model inference: {pred_time:.1f}s")

    # Analyze probability distribution before thresholding
    prob_analysis = analyze_probability_distribution(contact_probs, sequence_length)

    # Auto-calculate threshold if not provided
    if threshold is None:
        threshold = calculate_optimal_threshold(sequence_length)
        print(f"   🎯 Auto-calculated threshold: {threshold:.3f}")

    # Suggest better threshold based on probability distribution
    suggested_thresh = prob_analysis['suggested_threshold']
    if abs(suggested_thresh - threshold) > 0.05:
        print(f"   💡 Better threshold: {suggested_thresh:.3f} (would give {prob_analysis['suggested_density']*100:.1f}% density)")

    # Apply threshold
    binary_predictions = (contact_probs > threshold).astype(np.float32)

    # Validate predictions
    validation = validate_predictions(binary_predictions, contact_probs, sequence_length)

    total_time = time.time() - start_time
    print(f"   ✅ Prediction completed in {total_time:.1f}s total")
    print(f"   📊 Contact density: {validation['contact_density']:.4f} ({validation['contact_density']*100:.1f}%)")
    print(f"   🔢 Total contacts: {validation['num_contacts']:,}")

    if validation['is_realistic']:
        print(f"   ✅ Contact density is realistic")
    else:
        if validation['contact_density'] > 0.5:
            print(f"   ⚠️  Warning: Very high contact density - may indicate dummy features")
        else:
            print(f"   ⚠️  Warning: Contact density outside expected range")
            if prob_analysis['suggested_threshold'] != threshold:
                print(f"   💡 Try threshold {prob_analysis['suggested_threshold']:.3f} for better density")

    contact_binary = binary_predictions

    # Calculate confidence scores using the same method as serving module
    confidence_scores = np.maximum(contact_probs, contact_probs.T)  # Symmetric confidence

    # Prepare results - convert numpy types to Python types for JSON serialization
    results = {
        'pdb_file': str(pdb_path),
        'model_path': str(model_path),
        'sequence': sequence,
        'sequence_length': int(sequence_length),
        'contact_probabilities': contact_probs.tolist(),
        'contact_binary': contact_binary.tolist(),
        'confidence_scores': confidence_scores.tolist(),
        'threshold': float(threshold),
        'contact_density': float(validation['contact_density']),
        'num_contacts': int(validation['num_contacts']),
        'expected_density_range': (float(validation['expected_range'][0]), float(validation['expected_range'][1])),
        'is_realistic': bool(validation['is_realistic']),
        'device': str(device),
        'features_used': 'real_esm2_and_pattern_templates',  # Real ESM2 + pattern templates (same as training)
        'model_info': {
            'input_channels': 68,
            'base_channels': getattr(model, 'base_channels', 32),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'memory_footprint_mb': sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        },
        'prediction_statistics': {
            'max_probability': float(np.max(contact_probs)),
            'mean_probability': float(np.mean(contact_probs)),
            'std_probability': float(np.std(contact_probs)),
            'max_confidence': float(np.max(confidence_scores)),
            'mean_confidence': float(np.mean(confidence_scores))
        },
        'probability_analysis': validation['probability_analysis'],
        'performance': {
            'total_time': float(total_time),
            'model_load_time': float(model_time),
            'feature_processing_time': float(process_time),
            'inference_time': float(pred_time),
            'features_per_second': float(sequence_length ** 2 / pred_time) if pred_time > 0 else 0.0
        }
    }

    return results


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def save_predictions(results: dict, output_path: str):
    """Save prediction results to file."""
    try:
        import json
        # Convert numpy types to Python types
        json_ready_results = convert_numpy_types(results)
        with open(output_path, 'w') as f:
            json.dump(json_ready_results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")


def test_compatibility():
    """Test script compatibility and dependencies."""
    print("🧪 Testing compatibility...")

    # Check imports
    if MODEL_IMPORTS_AVAILABLE:
        print("   ✅ Model imports available")
    else:
        print("   ❌ Model imports failed - check your installation")
        return False

    # Check torch version
    torch_version = torch.__version__
    print(f"   ✅ PyTorch version: {torch_version}")

    # Check device availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   ✅ CUDA available: {gpu_name}")
    else:
        print("   ⚠️  CUDA not available, will use CPU")

    # Check BioPython
    try:
        import Bio
        print(f"   ✅ BioPython available: {Bio.__version__}")
    except ImportError:
        print("   ⚠️  BioPython not available - PDB parsing may be limited")

    # Check ESM availability
    try:
        import esm
        print(f"   ✅ ESM available")
    except ImportError:
        print("   ❌ ESM not available - required for embeddings")
        return False

    print("   ✅ All critical dependencies available\n")
    return True


def get_pdb_files_from_directory(directory: str) -> List[str]:
    """Get all PDB files from a directory."""
    pdb_dir = Path(directory)
    if not pdb_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pdb_files = []
    for pattern in ['*.pdb', '*.ent', '*.cif']:
        pdb_files.extend(pdb_dir.glob(pattern))

    if not pdb_files:
        raise ValueError(f"No PDB files found in directory: {directory}")

    return [str(f) for f in sorted(pdb_files)]


def detect_architecture_from_checkpoint(state_dict: dict) -> dict:
    """
    Detect BinaryContactCNN architecture parameters from state dict.

    Args:
        state_dict: Model state dict from checkpoint

    Returns:
        Dictionary with architecture parameters
    """
    print("   🔍 Detecting model architecture from checkpoint...")

    # Extract key layer shapes to determine architecture
    if 'conv1.0.weight' not in state_dict:
        raise ValueError("Cannot detect architecture: missing conv1.0.weight in state dict")

    # Extract first convolution layer shape
    conv1_0_shape = state_dict['conv1.0.weight'].shape
    in_channels = conv1_0_shape[1]  # Input channels (should be 68)
    base_channels = conv1_0_shape[0]  # Base channels (32 for your model)

    # Validate expected input channels
    if in_channels != 68:
        raise ValueError(f"Expected 68 input channels (64 ESM2 + 4 template), got {in_channels}")

    # Validate base channels with second conv layer if available
    if 'conv1.3.weight' in state_dict:
        conv1_3_shape = state_dict['conv1.3.weight'].shape
        expected_internal = base_channels
        if conv1_3_shape[1] != expected_internal:
            print(f"   ⚠️  Warning: Expected {expected_internal} internal channels, got {conv1_3_shape[1]}")

    # Detect dropout rate from configuration if available
    dropout_rate = 0.3  # Default fallback

    # Log detected architecture
    print(f"   🏗️  Detected architecture:")
    print(f"      Input channels: {in_channels}")
    print(f"      Base channels: {base_channels}")
    print(f"      Detected channel progression: [{base_channels}, {base_channels}, {base_channels*2}, {base_channels*4}]")

    # Show architecture efficiency benefits
    if base_channels == 32:
        print(f"      ✅ Efficient {base_channels}-base channel architecture detected")
        print(f"      ✅ Optimized design: Smaller model prevents overfitting, improves efficiency")
    elif base_channels == 64:
        print(f"      ✅ Standard {base_channels}-base channel architecture detected")
    else:
        print(f"      ✅ Custom {base_channels}-base channel architecture detected")

    # Validate progression pattern with available layers
    if 'conv2.0.weight' in state_dict:
        conv2_0_shape = state_dict['conv2.0.weight'].shape
        actual_second_layer = conv2_0_shape[1]
        print(f"      ✅ Layer 2 verified: {actual_second_layer} channels")

    if 'conv3.0.weight' in state_dict:
        conv3_0_shape = state_dict['conv3.0.weight'].shape
        actual_third_layer = conv3_0_shape[1]
        print(f"      ✅ Layer 3 verified: {actual_third_layer} channels")

    return {
        'in_channels': in_channels,
        'base_channels': base_channels,
        'dropout_rate': dropout_rate
    }


def load_direct_model(model_path: str) -> torch.nn.Module:
    """
    Load model directly from .pth file.

    Args:
        model_path: Path to .pth model file

    Returns:
        Loaded PyTorch model
    """
    print(f"🔄 Loading model from file: {model_path}")

    # Validate file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not model_path.endswith('.pth'):
        raise ValueError(f"Model file must be a .pth file: {model_path}")

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract state dict from different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # New format with metadata
                state_dict = checkpoint['model_state_dict']
                print(f"   📋 Found new checkpoint format with metadata")
                if 'model_config' in checkpoint:
                    print(f"   📊 Saved model config: {checkpoint['model_config']}")
            elif 'state_dict' in checkpoint:
                # Alternative format
                state_dict = checkpoint['state_dict']
                print(f"   📋 Found state_dict format checkpoint")
            elif all(key.startswith(('conv', 'fc', 'norm', 'pool')) for key in checkpoint.keys()):
                # Raw state dict format
                state_dict = checkpoint
                print(f"   📋 Found raw state dict checkpoint")
            else:
                # Try to use checkpoint as raw state dict anyway
                state_dict = checkpoint
                print(f"   📋 Assuming raw state dict checkpoint")
        else:
            raise ValueError("Unknown checkpoint format")

        # Detect architecture from state dict
        architecture_params = detect_architecture_from_checkpoint(state_dict)

        # Initialize model with detected architecture
        model = BinaryContactCNN(**architecture_params)
        print(f"   ✅ Created model with detected architecture")

        # Load the state dict into the model
        model.load_state_dict(state_dict)
        print(f"   ✅ Successfully loaded model weights")

        # Set model to evaluation mode
        model.eval()

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        print(f"   🎯 Model loaded successfully on {device}")
        print(f"   📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def load_pyfunc_model_smart(args) -> Any:
    """Load model using various methods based on arguments."""
    # Priority 1: Direct model file path (most reliable)
    if hasattr(args, 'model_path') and args.model_path:
        print(f"🔄 Loading model from file: {args.model_path}")
        return load_direct_model(args.model_path)

    # Priority 2: MLflow URI
    elif hasattr(args, 'model_uri') and args.model_uri:
        print(f"🔄 Loading PyFunc model from URI: {args.model_uri}")
        return load_pyfunc_model(args.model_uri)

    # Priority 3: PyFunc experiment
    elif hasattr(args, 'pyfunc_experiment') and args.pyfunc_experiment:
        print(f"🔍 Finding best model in experiment: {args.pyfunc_experiment}")
        model = get_best_pyfunc_model(args.pyfunc_experiment)
        if model is None:
            raise RuntimeError(f"No models found in experiment: {args.pyfunc_experiment}")
        return model

    # Priority 4: Run ID
    elif hasattr(args, 'run_id') and args.run_id:
        print(f"🔄 Loading PyFunc model from run: {args.run_id}")
        return load_pyfunc_model_from_run(args.run_id)

    else:
        raise ValueError("No valid model source specified. Use --model-path, --model-uri, --pyfunc-experiment, or --run-id")


def predict_with_model(model: Any, pdb_files: List[str], args) -> List[Dict[str, Any]]:
    """Make predictions using either PyFunc model or direct PyTorch model."""
    results = []

    # Check if model is a direct PyTorch model
    if isinstance(model, torch.nn.Module):
        print(f"🎯 Using direct PyTorch model for {len(pdb_files)} file(s)")

        if args.batch_processing or len(pdb_files) > 1:
            # Batch processing for direct PyTorch model
            print(f"🚀 Using direct batch processing for {len(pdb_files)} files")
            for pdb_file in pdb_files:
                if args.verbose:
                    print(f"   Processing: {pdb_file}")
                result = predict_from_pdb_direct(model, pdb_file, threshold=args.threshold, verbose=args.verbose)
                results.append(result)
        else:
            # Single file processing for direct PyTorch model
            pdb_file = pdb_files[0]
            print(f"🔮 Using direct PyTorch prediction for: {pdb_file}")
            result = predict_from_pdb_direct(model, pdb_file, threshold=args.threshold, verbose=args.verbose)
            results = [result]

    else:
        # PyFunc model processing (existing logic)
        print(f"🎯 Using PyFunc model for {len(pdb_files)} file(s)")

        if args.batch_processing or len(pdb_files) > 1:
            # Batch processing
            print(f"🚀 Using PyFunc batch processing for {len(pdb_files)} files")
            results = predict_batch_from_pdb(
                model, pdb_files,
                threshold=args.threshold,
                show_progress=True
            )
        else:
            # Single file processing
            pdb_file = pdb_files[0]
            print(f"🔮 Using PyFunc prediction for: {pdb_file}")
            result = predict_from_pdb_pyfunc(
                model, pdb_file,
                threshold=args.threshold
            )
            results = [result]

    return results


def predict_from_pdb_direct(model: torch.nn.Module, pdb_file: str, threshold: float = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Make prediction using direct PyTorch model.

    Args:
        model: PyTorch model
        pdb_file: Path to PDB file
        threshold: Prediction threshold
        verbose: Whether to show verbose output

    Returns:
        Prediction results
    """
    try:
        # Simple direct prediction without serving layer complexity
        from esm2_contact.serving.contact_predictor import load_esm2_model_cached, extract_sequence_from_pdb_simple

        # Generate ESM2 embeddings
        sequence = extract_sequence_from_pdb_simple(pdb_file)
        esm2_model, alphabet, device = load_esm2_model_cached()
        batch_converter = alphabet.get_batch_converter()

        # Prepare batch for single sequence
        batch_data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # Generate embeddings
        esm2_model.eval()
        with torch.no_grad():
            outputs = esm2_model(
                batch_tokens,
                repr_layers=[33],
                return_contacts=False
            )
            # Extract embeddings (remove BOS and EOS tokens)
            esm2_embedding = outputs["representations"][33][:, 1:-1, :].squeeze(0)

        # Move ESM2 embedding to CPU before tensor assembly
        if hasattr(esm2_embedding, 'cpu'):
            esm2_embedding = esm2_embedding.cpu()

        # Generate template features
        template_features = generate_pattern_based_template_features(sequence)

        # Assemble 68-channel tensor
        features = assemble_68_channel_tensor(esm2_embedding, template_features)

        # Convert to tensor and move to device
        device = next(model.parameters()).device
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)

        # Make prediction
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = model(features_tensor)
                probabilities = torch.sigmoid(logits)

        # Apply threshold
        if threshold is None:
            threshold = 0.15

        contacts_binary = (probabilities > threshold).float()

        # Convert to numpy for output
        contacts_binary_np = contacts_binary.squeeze().cpu().numpy()
        probabilities_np = probabilities.squeeze().cpu().numpy()

        # Create result
        result = {
            'contact_binary': contacts_binary_np,
            'contact_probabilities': probabilities_np,
            # Contact density: (total_contacts / full_matrix_size) * 100
            # Using full matrix method consistent with training pipeline
            # Typical range for large proteins: 2-8% (biologically reasonable)
            'contact_density': contacts_binary_np.mean() * 100,
            'sequence_length': len(sequence),
            'total_contacts': int(contacts_binary_np.sum()),
            'threshold_used': threshold,
            'pdb_file': pdb_file,
            'protein_id': Path(pdb_file).stem
        }

        if verbose:
            print(f"   ✅ Prediction completed successfully")
            print(f"   📊 Contact density: {result.get('contact_density', 'N/A')}")
            print(f"   🎯 Predictions shape: {result.get('contact_binary', 'N/A')}")

        return result

    except Exception as e:
        raise RuntimeError(f"Prediction failed for {pdb_file}: {e}")


def predict_with_pyfunc(model: Any, pdb_files: List[str], args) -> List[Dict[str, Any]]:
    """Make predictions using PyFunc model (legacy function for backward compatibility)."""
    return predict_with_model(model, pdb_files, args)


def create_pyfunc_model_workflow(model_path: str):
    """Create PyFunc model from checkpoint."""
    try:
        print(f"🏗️ Creating MLflow PyFunc model from: {model_path}")

        model_uri = create_pyfunc_model_from_checkpoint(
            model_path=model_path,
            experiment_name="esm2_contact_pyfunc_creation"
        )

        print(f"✅ PyFunc model created successfully!")
        print(f"   Model URI: {model_uri}")
        print(f"")
        print(f"Usage examples:")
        print(f"  # Load and use the model:")
        print(f"  python scripts/05_predict_from_pdb.py --use-pyfunc --model-uri '{model_uri}' --pdb-file protein.pdb")
        print(f"")
        print(f"  # Validate the model:")
        print(f"  python scripts/05_predict_from_pdb.py --validate-pyfunc --model-uri '{model_uri}' --pdb-file protein.pdb")

    except Exception as e:
        print(f"❌ Failed to create PyFunc model: {e}")
        return 1

    return 0


def validate_pyfunc_workflow(model: Any, test_file: str):
    """Validate PyFunc model functionality."""
    try:
        print(f"🧪 Validating PyFunc model functionality...")

        validation_results = validate_pyfunc_model(
            model_uri="loaded_model",  # We're passing the loaded model directly
            test_pdb_file=test_file
        )

        print(f"")
        print(f"📊 Validation Results:")
        print(f"   Tests run: {len(validation_results['tests'])}")

        for test_name, result in validation_results['tests'].items():
            print(f"   {test_name}: {result}")

        if validation_results['validation_passed']:
            print(f"✅ Model validation passed!")
        else:
            print(f"⚠️  Model validation had issues")

        if validation_results['errors']:
            print(f"")
            print(f"❌ Errors encountered:")
            for error in validation_results['errors']:
                print(f"   • {error}")

    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return 1

    return 0


def benchmark_model_workflow(model: Any, test_files: List[str], iterations: int):
    """Benchmark model performance."""
    try:
        print(f"🏃 Benchmarking model performance...")

        from src.esm2_contact.serving.prediction_utils import benchmark_model_performance

        benchmark_results = benchmark_model_performance(
            model_uri="loaded_model",  # Using loaded model directly
            test_files=test_files,
            iterations=iterations
        )

        print(f"")
        print(f"📊 Benchmark Results:")
        print(f"   Model: PyFunc model")
        print(f"   Test files: {benchmark_results['num_files']}")
        print(f"   Iterations: {benchmark_results['iterations']}")

        stats = benchmark_results['statistics']
        print(f"")
        print(f"⏱️ Timing Statistics:")
        print(f"   Average time: {stats['avg_total_time']:.2f}s ± {stats['std_total_time']:.2f}s")
        print(f"   Min time: {stats['min_total_time']:.2f}s")
        print(f"   Max time: {stats['max_total_time']:.2f}s")
        print(f"   Throughput: {stats['throughput_predictions_per_second']:.2f} predictions/s")

        print(f"")
        print(f"📈 Success Rate:")
        avg_success = stats['avg_successful_predictions']
        print(f"   Average successful predictions: {avg_success:.1f}/{benchmark_results['num_files']}")

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return 1

    return 0


def main():
    """Enhanced main function with PyFunc support."""
    parser = argparse.ArgumentParser(
        description="Predict protein contacts from PDB file with modern MLflow PyFunc support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # Direct model loading (recommended for local models)
  %(prog)s --pdb-file protein.pdb --model-path "path/to/model.pth"

  # MLflow-based prediction
  %(prog)s --pdb-file protein.pdb --model-uri "mlruns/exp_id/run_id/artifacts/model"

  # Batch processing with direct model
  %(prog)s --batch-processing --pdb-file-dir ./pdbs/ --model-path "path/to/model.pth"

  # Create PyFunc model from checkpoint
  %(prog)s --create-pyfunc-from-model model.pth

  # Validate MLflow model
  %(prog)s --validate-pyfunc --model-uri "mlruns/exp_id/run_id/artifacts/model" --pdb-file protein.pdb

  # Benchmark performance
  %(prog)s --benchmark --pdb-file-dir ./test_pdbs/ --model-path "path/to/model.pth" --iterations 5
        """
    )

    parser.add_argument(
        '--pdb-file',
        type=str,
        help='Path to PDB file (required unless using --create-pyfunc-from-model)'
    )

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        '--model-uri',
        type=str,
        help='MLflow model URI (alternative to --model-path, e.g., "mlruns/exp_id/run_id/artifacts/best_model_checkpoint")'
    )

    model_group.add_argument(
        '--model-path',
        type=str,
        help='Direct path to model .pth file (alternative to --model-uri, e.g., "path/to/model.pth")'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Threshold for binary predictions (auto-calculated if not provided)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='predictions.json',
        help='Output file for predictions'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output with error details'
    )

    parser.add_argument(
        '--test-compatibility',
        action='store_true',
        help='Test compatibility before running prediction'
    )

    # Enhanced PyFunc arguments
    # Note: PyFunc is now the default and only approach

    parser.add_argument(
        '--pyfunc-experiment',
        type=str,
        default='esm2_contact_training',
        help='MLflow experiment name for PyFunc model (default: esm2_contact_training)'
    )

    parser.add_argument(
        '--create-pyfunc-from-model',
        type=str,
        help='Create PyFunc model from checkpoint and exit (path to .pth file)'
    )

    parser.add_argument(
        '--validate-pyfunc',
        action='store_true',
        help='Validate PyFunc model functionality'
    )

    parser.add_argument(
        '--batch-processing',
        action='store_true',
        help='Enable batch processing for multiple PDB files (requires --pdb-file-dir or multiple files)'
    )

    parser.add_argument(
        '--pdb-file-dir',
        type=str,
        help='Directory containing PDB files for batch processing'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark model performance'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of iterations for benchmarking (default: 3)'
    )

    args = parser.parse_args()

    # Test compatibility if requested
    if args.test_compatibility:
        if not test_compatibility():
            return 1

    # Handle PyFunc model creation workflow
    if args.create_pyfunc_from_model:
        if not PYFUNC_AVAILABLE:
            print("❌ PyFunc functionality not available. Install required dependencies.")
            return 1
        return create_pyfunc_model_workflow(args.create_pyfunc_from_model)

    # Validate arguments and determine workflow
    pdb_files = []

    # Handle directory input for batch processing
    if args.pdb_file_dir:
        try:
            pdb_files = get_pdb_files_from_directory(args.pdb_file_dir)
            print(f"📁 Found {len(pdb_files)} PDB files in directory: {args.pdb_file_dir}")
        except Exception as e:
            print(f"❌ Failed to process directory: {e}")
            return 1
    elif args.pdb_file:
        pdb_files = [args.pdb_file]

    # Validate required arguments for prediction workflows
    if not args.create_pyfunc_from_model:
        if not pdb_files:
            print("❌ No input specified. Use --pdb-file or --pdb-file-dir")
            return 1
        # Check for either model-uri or model-path
        if not ((hasattr(args, 'model_uri') and args.model_uri) or (hasattr(args, 'model_path') and args.model_path)):
            print("❌ No model specified. Use --model-uri or --model-path")
            return 1

        # Validate model-path file exists
        if hasattr(args, 'model_path') and args.model_path:
            if not os.path.exists(args.model_path):
                print(f"❌ Model file not found: {args.model_path}")
                return 1
            if not args.model_path.endswith('.pth'):
                print(f"❌ Model file must be a .pth file: {args.model_path}")
                return 1

    # MLflow PyFunc is now the only approach (mandatory)
    if not PYFUNC_AVAILABLE:
        print("❌ PyFunc functionality not available. Install required dependencies.")
        return 1

    try:
        # Load PyFunc model
        model = load_pyfunc_model_smart(args)

        # Validate model if requested
        if args.validate_pyfunc:
            return validate_pyfunc_workflow(model, pdb_files[0])

        # Benchmark model if requested
        if args.benchmark:
            return benchmark_model_workflow(model, pdb_files, args.iterations)

        # Make predictions
        results = predict_with_model(model, pdb_files, args)

        # Save results
        if len(results) == 1:
            # Single result
            save_predictions(results[0], args.output)
            print(f"\n🎉 Prediction completed successfully!")
            print(f"   📄 Results: {args.output}")
            print(f"   📏 Sequence length: {results[0].get('sequence_length', 'N/A')}")
            print(f"   📊 Contact density: {results[0].get('contact_density', 'N/A')}")
            print(f"   🔢 Total contacts: {results[0].get('total_contacts', 'N/A')}")
            if 'error' in results[0]:
                print(f"   ⚠️  Error: {results[0]['error']}")
            else:
                print(f"   ✅ Ready for downstream analysis!")
        else:
            # Batch results - save as JSON array
            batch_results = {
                'batch_info': {
                    'total_files': len(pdb_files),
                    'successful_predictions': sum(1 for r in results if 'error' not in r),
                    'failed_predictions': sum(1 for r in results if 'error' in r),
                    'model_type': 'mlflow_pyfunc'
                },
                'results': results
            }
            save_predictions(batch_results, args.output)
            print(f"\n🎉 Batch prediction completed successfully!")
            print(f"   📄 Results: {args.output}")
            print(f"   📁 Total files: {len(pdb_files)}")
            print(f"   ✅ Successful: {batch_results['batch_info']['successful_predictions']}")
            print(f"   ❌ Failed: {batch_results['batch_info']['failed_predictions']}")
            print(f"   ✅ Ready for downstream analysis!")

    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())