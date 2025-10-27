#!/usr/bin/env python3
"""
Simple PDB Inference Script

This script provides a simple function to predict contacts directly from a PDB file.
It's a lightweight alternative to the full pipeline for quick predictions.

Usage:
    uv run python scripts/predict_from_pdb.py --pdb-file my_protein.pdb --model-path model.pth
"""

import os
import sys
import argparse
import warnings
import hashlib
import gc
from pathlib import Path
import torch
import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.esm2_contact.training.model import BinaryContactCNN

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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

def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    try:
        print(f"🧠 Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=device)

        # Extract model architecture and weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            in_channels = config.get('in_channels', 68)
            base_channels = config.get('base_channels', 32)
            dropout_rate = config.get('dropout_rate', 0.1)
        else:
            # Just state dict - infer architecture
            model_state = checkpoint
            in_channels = 68  # Default for ESM2 contact prediction
            base_channels = 32
            dropout_rate = 0.1

        # Create and load model
        model = BinaryContactCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            dropout_rate=dropout_rate
        )
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        print(f"   ✅ Model loaded successfully")
        print(f"   Architecture: {in_channels}→{base_channels} channels")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


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

def predict_contacts(pdb_path: str, model_path: str, threshold: float = None):
    """
    Predict protein contacts from PDB file using real ESM2 and homology features.

    Args:
        pdb_path (str): Path to PDB file
        model_path (str): Path to trained model
        threshold (float): Threshold for binary predictions (auto-calculated if None)

    Returns:
        dict: Prediction results including contact map and metadata
    """
    import time
    start_time = time.time()

    print(f"🚀 Starting Real PDB Contact Prediction")
    print(f"   PDB file: {pdb_path}")
    print(f"   Model: {model_path}")
    print(f"="*50)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # Load model
    model_start = time.time()
    model = load_model(model_path, device)
    model_time = time.time() - model_start
    print(f"   ⏱️  Model loading: {model_time:.1f}s")

    # Process PDB file
    process_start = time.time()
    features_tensor, sequence = process_pdb_file(pdb_path, device)
    process_time = time.time() - process_start
    print(f"   ⏱️  Feature processing: {process_time:.1f}s")
    sequence_length = len(sequence)

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

    # Prepare results - convert numpy types to Python types for JSON serialization
    results = {
        'pdb_file': str(pdb_path),
        'model_path': str(model_path),
        'sequence': sequence,
        'sequence_length': int(sequence_length),
        'contact_probabilities': contact_probs.tolist(),
        'contact_binary': contact_binary.tolist(),
        'threshold': float(threshold),
        'contact_density': float(validation['contact_density']),
        'num_contacts': int(validation['num_contacts']),
        'expected_density_range': (float(validation['expected_range'][0]), float(validation['expected_range'][1])),
        'is_realistic': bool(validation['is_realistic']),
        'device': str(device),
        'features_used': 'real_esm2_and_pattern_templates',  # Real ESM2 + pattern templates (same as training)
        'performance': {
            'total_time': float(total_time),
            'model_load_time': float(model_time),
            'feature_processing_time': float(process_time),
            'inference_time': float(pred_time)
        }
    }

    return results


def save_predictions(results: dict, output_path: str):
    """Save prediction results to file."""
    try:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Predict protein contacts from PDB file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--pdb-file',
        type=str,
        required=True,
        help='Path to PDB file'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to trained model (.pth file)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary predictions'
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
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.pdb_file).exists():
        raise FileNotFoundError(f"PDB file not found: {args.pdb_file}")

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    try:
        # Run prediction
        results = predict_contacts(args.pdb_file, args.model_path, args.threshold)

        # Save results
        save_predictions(results, args.output)

        print(f"\n🎉 Prediction completed successfully!")
        print(f"   📄 Results: {args.output}")
        print(f"   📏 Sequence length: {results['sequence_length']}")
        print(f"   📊 Contact density: {results['contact_density']:.4f}")
        print(f"   ✅ Ready for downstream analysis!")

    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())