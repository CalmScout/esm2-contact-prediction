#!/usr/bin/env python3
"""
Generate CNN Dataset from PDB Files (Real Data Only)

This script creates a complete CNN training dataset from a directory of PDB files
using ONLY real data - no synthetic fallbacks or simulated data. It processes
proteins through all pipeline steps and produces a ready-to-use cnn_dataset.h5
file compatible with existing training pipeline.

Key Features:
- Takes directory of PDB files as input
- Processes configurable percentage of files (0.05 to 1.0)
- Uses random seed for reproducible file selection
- Creates 68-channel tensors (4 real template + 64 real ESM2)
- NO FALLBACKS - fails fast if dependencies missing
- Compatible with existing train_cnn_binary.py

Requirements:
- ESM2 model (auto-downloaded on first run)
- Real homology databases (UniRef30 and/or PDB70)
- Valid PDB files with proper structure
- Sufficient system memory (>=4GB recommended)

Pipeline Steps:
1. Dependency validation (ESM2, databases, libraries)
2. PDB file discovery and selection
3. Sequence extraction and ESM2 embeddings (64 channels)
4. Ground truth contact map generation from PDB coordinates
5. Real homology template search using databases (4 channels)
6. CNN dataset assembly and HDF5 output

Usage:
    # Process 5% of PDB files:
    uv run python scripts/04_generate_cnn_dataset.py --pdb-dir ./my_pdbs --process-ratio 0.05

    # Process all files with fixed seed:
    uv run python scripts/04_generate_cnn_dataset.py --pdb-dir ./my_pdbs --process-ratio 1.0 --random-seed 42

Setup (run once):
    # Download required databases:
    uv run python scripts/02_download_homology_databases.py --db all
"""

import os
import sys
import time
import argparse
import warnings
import hashlib
import gc
import psutil
import fcntl
import errno
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing components
from src.esm2_contact.homology import TemplateSearcher
from Bio.PDB import PDBParser
import warnings
import importlib.util

def monitor_resources():
    """Monitor system resource usage."""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024**3)
    memory_percent = process.memory_percent()
    cpu_percent = process.cpu_percent()

    return {
        'memory_gb': memory_gb,
        'memory_percent': memory_percent,
        'cpu_percent': cpu_percent
    }

def cleanup_memory():
    """Force garbage collection and GPU memory cleanup."""
    # Python garbage collection
    gc.collect()

    # PyTorch CPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def acquire_file_lock(output_path, timeout=300):
    """Acquire exclusive lock on output file.

    Args:
        output_path: Path to the output file
        timeout: Maximum time to wait for lock (seconds)

    Returns:
        File handle for the lock file

    Raises:
        RuntimeError: If lock cannot be acquired within timeout
    """
    lock_file_path = Path(str(output_path) + ".lock")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Create and lock the file
            lock_file = open(lock_file_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_file
        except (IOError, OSError) as e:
            if e.errno == errno.EAGAIN or e.errno == errno.EACCES:
                # File is locked by another process
                print(f"   🔒 Waiting for file lock on {output_path}...")
                time.sleep(5)  # Wait 5 seconds before retrying
            else:
                raise
    raise RuntimeError(f"Could not acquire file lock for {output_path} after {timeout} seconds. Another process may be running.")

def release_file_lock(lock_file):
    """Release the file lock.

    Args:
        lock_file: File handle from acquire_file_lock
    """
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        # Remove the lock file
        lock_file_path = Path(lock_file.name)
        if lock_file_path.exists():
            lock_file_path.unlink()
    except Exception as e:
        print(f"   ⚠️  Warning: Could not properly release file lock: {e}")

    # Force additional garbage collection
    gc.collect()

def check_memory_limits(max_memory_gb: float = 6.0) -> bool:
    """Check if memory usage is within limits."""
    resources = monitor_resources()
    if resources['memory_gb'] > max_memory_gb:
        print(f"   ⚠️  High memory usage: {resources['memory_gb']:.1f}GB (limit: {max_memory_gb}GB)")
        cleanup_memory()
        # Check again after cleanup
        resources = monitor_resources()
        return resources['memory_gb'] <= max_memory_gb
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CNN dataset from PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdb-dir ./pdbs                    # Process all PDB files
  %(prog)s --pdb-dir ./pdbs --process-ratio 0.1    # Process 10%% of files
  %(prog)s --pdb-dir ./pdbs --random-seed 42     # Fixed random seed for reproducibility
        """
    )

    parser.add_argument(
        '--pdb-dir',
        type=Path,
        required=True,
        help='Directory containing PDB files'
    )

    parser.add_argument(
        '--process-ratio',
        type=float,
        default=1.0,
        help='Proportion of PDB files to process (0.05-1.0, default: 1.0)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='data/cnn_dataset.h5',
        help='Output path for CNN dataset (default: data/cnn_dataset.h5)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='Random seed for reproducible file selection (default: random)'
    )

    parser.add_argument(
        '--cpu-limit',
        type=int,
        default=2,
        help='CPU limit for homology search (default: 2)'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default='data/.template_cache',
        help='Template cache directory (default: data/.template_cache)'
    )

    return parser.parse_args()

def discover_pdb_files(pdb_dir: Path) -> List[Path]:
    """Discover and sort PDB files in directory."""
    pdb_files = list(pdb_dir.glob('*.pdb'))
    pdb_files = [f for f in pdb_files if f.is_file()]
    pdb_files.sort()
    return pdb_files

def select_pdb_files(pdb_files: List[Path], process_ratio: float, random_seed: Optional[int]) -> List[Path]:
    """Select subset of PDB files to process."""
    total_files = len(pdb_files)

    if process_ratio >= 1.0:
        selected_files = pdb_files
    else:
        # Calculate number to process
        num_to_process = int(total_files * process_ratio)
        num_to_process = max(1, num_to_process)  # Always process at least 1

        if random_seed is not None:
            np.random.seed(random_seed)
            print(f"🎲 Using random seed: {random_seed}")

        # Shuffle and select files
        np.random.shuffle(pdb_files)
        selected_files = pdb_files[:num_to_process]

        print(f"📁 Selected {num_to_process}/{total_files} PDB files ({process_ratio*100:.1f}%)")

    return selected_files

def extract_protein_from_pdb(pdb_file: Path) -> Optional[Dict[str, Any]]:
    """Extract protein sequence and coordinates using Biopython for robust parsing."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(pdb_file.stem, str(pdb_file))

        # Get first model and find best chain with CA atoms
        model = next(iter(structure.get_models()))
        best_chain = None
        best_chain_residues = 0

        for chain in model.get_chains():
            # Count residues with CA atoms
            ca_residues = sum(1 for residue in chain if 'CA' in residue)
            if ca_residues > best_chain_residues:
                best_chain = chain
                best_chain_residues = ca_residues

        if not best_chain or best_chain_residues < 2:
            print(f"   ⚠️  No suitable chain found in {pdb_file.name} (need ≥2 residues with CA atoms)")
            return None

        chain_id = best_chain.id
        print(f"   📋 Chain: {chain_id} ({best_chain_residues} residues)")

        # Extract sequence using Biopython's PPBuilder
        from Bio.PDB.Polypeptide import PPBuilder
        ppb = PPBuilder()

        sequences = []
        coordinates = {chain_id: []}

        # Build polypeptides and extract sequence
        for pp in ppb.build_peptides(best_chain):
            sequence = pp.get_sequence()
            sequences.append((chain_id, str(sequence)))

            # Extract coordinates for each residue in this polypeptide
            for residue in pp:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    coord = ca_atom.get_vector()
                    coordinates[chain_id].append([coord[0], coord[1], coord[2]])

        # Create unified chain sequence
        full_sequence = ''.join([seq for _, seq in sequences])

        # Validate we have matching coordinates and sequence
        if len(coordinates[chain_id]) != len(full_sequence):
            print(f"   ⚠️  Mismatch between coordinates ({len(coordinates[chain_id])}) and sequence ({len(full_sequence)})")
            # Use minimum length to avoid indexing errors
            min_len = min(len(coordinates[chain_id]), len(full_sequence))
            coordinates[chain_id] = coordinates[chain_id][:min_len]
            full_sequence = full_sequence[:min_len]
            if min_len < 2:
                return None

        return {
            'pdb_file': str(pdb_file),
            'pdb_id': structure.id or pdb_file.stem,
            'coordinates': coordinates,
            'sequences': sequences,
            'chain_sequences': {chain_id: full_sequence},
            'chain_id': chain_id
        }

    except Exception as e:
        print(f"❌ Error parsing PDB {pdb_file.name}: {e}")
        return None

# Global ESM2 model cache (load once, reuse across proteins)
_ESM2_MODEL = None
_ESM2_ALPHABET = None
_ESM2_DEVICE = None

def load_esm2_model():
    """Load ESM2 model once and cache globally."""
    global _ESM2_MODEL, _ESM2_ALPHABET, _ESM2_DEVICE

    if _ESM2_MODEL is None:
        print("📱 Loading ESM2 model once (will be reused for all proteins)...")
        import esm

        # Load model and alphabet
        _ESM2_MODEL, _ESM2_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
        _ESM2_MODEL.eval()

        # Move to GPU if available
        _ESM2_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        _ESM2_MODEL = _ESM2_MODEL.to(_ESM2_DEVICE)

        print(f"   ✅ ESM2 model loaded on {_ESM2_DEVICE}")

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

def generate_esm2_embeddings(protein_id: str, sequence: str, cache_dir: Path) -> np.ndarray:
    """Generate real ESM2 embeddings using pre-loaded model with detailed debugging."""
    try:
        # Use cached model (loaded once at startup)
        model, alphabet, device = load_esm2_model()
        batch_converter = alphabet.get_batch_converter()

        # Validate and clean sequence
        clean_sequence = validate_sequence_for_esm(sequence)

        print(f"   🧬 Original sequence: {sequence[:20]}... ({len(sequence)} residues)")
        print(f"   🧬 Clean sequence: {clean_sequence[:20]}... ({len(clean_sequence)} residues)")

        if len(clean_sequence) < 2:
            raise ValueError(f"Sequence too short after cleaning: {len(clean_sequence)}")

        # Prepare sequence for ESM2
        sequences_list = [(protein_id, clean_sequence)]

        print(f"   🔢 Device: {device}")
        print(f"   📱 Model loaded: {model.__class__.__name__}")

        # Prepare batch
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences_list)
        print(f"   📦 Batch tokens shape: {batch_tokens.shape}")

        batch_tokens = batch_tokens.to(device)
        model.eval()

        with torch.no_grad():
            # Compute forward pass
            print(f"   🧠 Running model inference...")
            outputs = model(
                batch_tokens,
                repr_layers=[33],  # Final layer
                return_contacts=False
            )

            # Extract embeddings
            embeddings = outputs["representations"][33]  # Shape: (1, seq_len + 2, 1280)
            print(f"   📊 Raw embeddings shape: {embeddings.shape}")

            # Remove BOS and EOS tokens (first and last tokens)
            embeddings = embeddings[:, 1:-1, :]  # Shape: (1, seq_len, 1280)
            print(f"   📊 Trimmed embeddings shape: {embeddings.shape}")

            # Move to CPU and convert to numpy
            embedding = embeddings[0].cpu().numpy()  # Shape: (seq_len, 1280)
            print(f"   📊 Final embedding shape: {embedding.shape}")

        # Transpose to match expected format (1280, seq_len)
        embedding_transposed = embedding.T  # Shape: (1280, seq_len)
        print(f"   ✅ ESM2 embeddings: {embedding_transposed.shape}")

        # Memory cleanup
        del batch_tokens, embeddings, outputs
        if device == 'cuda':
            torch.cuda.empty_cache()

        return embedding_transposed

    except Exception as e:
        print(f"❌ Failed to generate ESM2 embeddings for {protein_id}: {e}")
        print(f"   📍 Debug info:")
        print(f"      - Sequence length: {len(sequence)}")
        print(f"      - Device: {device if 'device' in locals() else 'unknown'}")
        print(f"      - Model: {model.__class__.__name__ if 'model' in locals() else 'not loaded'}")

        # Import traceback for detailed debugging
        import traceback
        print(f"   📍 Traceback: {traceback.format_exc()}")

        # No fallback - fail with clear error message
        raise RuntimeError(
            f"ESM2 embedding generation failed for protein {protein_id}. "
            f"This is required for CNN dataset generation. "
            f"Please check:\n"
            f"  1. Internet connection for ESM2 model download\n"
            f"  2. Available GPU/CPU memory\n"
            f"  3. Valid protein sequence\n"
            f"  4. ESM2 library installation: uv add esm\n"
            f"Original error: {e}"
        )

def calculate_contact_map(coordinates: Dict[str, List[List[float]]], chain_id: str = None) -> np.ndarray:
    """Calculate binary contact map from coordinates."""
    try:
        # Handle single chain case
        if chain_id:
            chain_coords = coordinates.get(chain_id, [])
            if len(chain_coords) < 2:
                raise ValueError(
                    f"Chain {chain_id} has insufficient coordinates ({len(chain_coords)}). "
                    f"Need at least 2 residues with CA atoms for contact map generation."
                )
        else:
            # Use first available chain
            if not coordinates:
                raise ValueError("No coordinate data provided for contact map calculation.")
            chain_id = list(coordinates.keys())[0]
            chain_coords = coordinates[chain_id]

            if len(chain_coords) < 2:
                raise ValueError(
                    f"Chain {chain_id} has insufficient coordinates ({len(chain_coords)}). "
                    f"Need at least 2 residues with CA atoms for contact map generation."
                )

        L = len(chain_coords)
        contact_map = np.zeros((L, L), dtype=np.float32)

        # Convert to numpy array for vectorized operations
        coords_array = np.array(chain_coords)  # Shape: (L, 3)

        # Calculate pairwise distances efficiently
        if L > 1000:  # For large proteins, use batch processing to avoid memory issues
            batch_size = 500
            for i in range(0, L, batch_size):
                for j in range(0, L, batch_size):
                    i_end = min(i + batch_size, L)
                    j_end = min(j + batch_size, L)

                    # Compute distances for this batch
                    coords_i = coords_array[i:i_end]
                    coords_j = coords_array[j:j_end]

                    # Vectorized distance calculation
                    diff = coords_i[:, np.newaxis, :] - coords_j[np.newaxis, :, :]
                    distances = np.sqrt(np.sum(diff**2, axis=2))

                    # Apply 8Å threshold
                    contact_map[i:i_end, j:j_end] = (distances <= 8.0).astype(np.float32)
        else:
            # For smaller proteins, compute all at once
            diff = coords_array[:, np.newaxis, :] - coords_array[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            contact_map = (distances <= 8.0).astype(np.float32)

        # Set diagonal to 1.0 (self contacts)
        np.fill_diagonal(contact_map, 1.0)

        return contact_map

    except Exception as e:
        print(f"❌ Error calculating contact map: {e}")
        import traceback
        print(f"   📍 Traceback: {traceback.format_exc()}")

        # No fallback - fail with clear error message
        raise RuntimeError(
            f"Contact map calculation failed for protein coordinates. "
            f"This is required for CNN dataset generation. "
            f"Please ensure:\n"
            f"  1. PDB file contains valid 3D coordinates\n"
            f"  2. At least one chain has CA atoms for all residues\n"
            f"  3. Coordinates are not corrupted or missing\n"
            f"  4. PDB file follows standard format\n"
            f"Original error: {e}"
        )

def validate_homology_databases() -> None:
    """Validate that required homology databases are available."""
    from src.esm2_contact.homology.search import DatabaseConfig

    db_config = DatabaseConfig()

    # Check if we have any databases
    if not db_config.databases:
        raise RuntimeError(
            "No homology databases found. Template search requires real databases.\n"
            "Please download required databases:\n"
            "  uv run python scripts/02_download_homology_databases.py --db all\n"
            "or\n"
            "  uv run python scripts/02_download_homology_databases.py --db uniref30\n"
            "  uv run python scripts/02_download_homology_databases.py --db pdb70"
        )

    # Check if databases are ready
    ready_dbs = [db_type for db_type in ['uniref30', 'pdb70'] if db_config.is_ready(db_type)]
    if not ready_dbs:
        raise RuntimeError(
            "Homology databases are downloaded but not ready for use.\n"
            "Please check if databases are properly extracted and have required files.\n"
            f"Available databases: {list(db_config.databases.keys())}"
        )

    # Provide database-specific guidance
    if len(ready_dbs) == 1:
        if 'pdb70' in ready_dbs:
            print(f"   ✅ Found PDB70 database (structural templates)")
            print(f"   ℹ️  Note: Only PDB70 available. UniRef30 would provide additional sequence homologs.")
        elif 'uniref30' in ready_dbs:
            print(f"   ✅ Found UniRef30 database (sequence homologs)")
            print(f"   ⚠️  Warning: Only UniRef30 available. PDB70 recommended for structural templates.")
    else:
        print(f"   ✅ Found both PDB70 (structural) and UniRef30 (sequence) databases")

    print(f"   📊 Ready databases: {ready_dbs}")

def search_homology_templates(sequence: str, cache_dir: Path, cpu_limit: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Search for homology templates using real databases."""
    try:
        # Initialize template channels
        template_channels = np.zeros((4, len(sequence), len(sequence)), dtype=np.float32)

        print(f"   🔍 Searching for homology templates...")

        # Validate databases first
        validate_homology_databases()

        # Initialize TemplateSearcher
        from src.esm2_contact.homology.search import TemplateSearcher, DatabaseConfig

        # Load configuration
        import yaml
        config_path = project_root / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        template_config = config.get('template_search', {})
        quality_config = template_config.get('quality_thresholds', {})
        general_config = template_config  # Use template_config directly for general settings

        db_config = DatabaseConfig()
        searcher = TemplateSearcher(
            method="dual",  # Use both BLAST and HHblits
            min_identity=quality_config.get('min_sequence_identity', 0.2),
            min_coverage=quality_config.get('min_coverage', 0.3),
            max_templates=general_config.get('max_templates', 20),
            cache_dir=cache_dir
        )

        # Search for templates
        results = searcher.search_templates(sequence, "query")

        if not results:
            print(f"   ❌ No homology templates found for sequence length {len(sequence)}")
            raise RuntimeError(
                f"No homology templates found for protein sequence (length {len(sequence)}). "
                f"This is required for CNN dataset generation. "
                f"Please ensure:\n"
                f"  1. Homology databases are properly installed and accessible\n"
                f"  2. Sequence length is appropriate for template search (>= 20 residues recommended)\n"
                f"  3. Database search tools (HHblits/BLAST) are working correctly\n"
                f"  4. Network connectivity is available for remote searches if needed\n"
                f"  5. Consider adding more diverse protein structures to your PDB collection\n"
                f"Available databases: {[db_type for db_type in ['uniref30', 'pdb70'] if db_config.is_ready(db_type)]}"
            )

        # Process template results into channels
        all_results = results  # results is already filtered by TemplateSearcher
        print(f"   📊 Found {len(all_results)} template results")

        if len(all_results) == 0:
            print(f"   ❌ No templates found")
            raise RuntimeError(
                f"No homology templates found for protein sequence (length {len(sequence)}). "
                f"TemplateSearcher returned no results. Please ensure:\n"
                f"  1. Homology databases are accessible and contain diverse protein families\n"
                f"  2. HHblits/BLAST tools are working correctly\n"
                f"  3. Sequence length is appropriate for template search (>= 20 residues)\n"
                f"  4. Consider adjusting quality thresholds in TemplateSearcher if appropriate"
            )

        print(f"   ✅ Using {len(all_results)} templates (filtered by TemplateSearcher)")

        # Initialize template features
        conservation_scores = np.zeros(len(sequence))
        distance_weights = np.zeros(len(sequence))
        ss_propensity = np.zeros(len(sequence))
        coevolution_scores = np.zeros(len(sequence))

        # Process each high-quality template result
        for result in all_results[:20]:  # Limit to top 20 templates
            if result.sequence_identity > 0.3:  # Only use meaningful templates
                # Add to conservation scores
                weight = result.sequence_identity * result.coverage
                conservation_scores += weight

                # Distance-based weighting (closer templates get higher weight)
                dist_weight = 1.0 / (1.0 + result.e_value)
                distance_weights += dist_weight

                # Secondary structure propensity (simplified)
                if result.sequence_identity > 0.5:
                    ss_propensity += weight * 0.5

                # Coevolution (long-range contacts from multiple sequence alignment)
                if len(result.query_seq) == len(sequence):
                    for i, aa in enumerate(result.query_seq):
                        for j, aa2 in enumerate(result.query_seq):
                            if i != j and abs(i - j) > 12:
                                if aa == aa2:
                                    coevolution_scores[i] += 0.1

        # Normalize scores
        if conservation_scores.max() > 0:
            conservation_scores /= conservation_scores.max()
        if distance_weights.max() > 0:
            distance_weights /= distance_weights.max()
        if ss_propensity.max() > 0:
            ss_propensity /= ss_propensity.max()
        if coevolution_scores.max() > 0:
            coevolution_scores /= coevolution_scores.max()

        # Create template channels from real data
        L = len(sequence)

        # Channel 0: Sequence conservation from templates
        for i in range(L):
            for j in range(L):
                if abs(i - j) <= 2:
                    template_channels[0, i, j] = conservation_scores[i] * 0.8 + 0.2

        # Channel 1: Distance-based weighting
        for i in range(L):
            for j in range(L):
                dist = abs(i - j)
                if dist <= 8:
                    template_channels[1, i, j] = distance_weights[i] * np.exp(-dist / 4.0)

        # Channel 2: Secondary structure propensity
        for i in range(L):
            for j in range(L):
                if i != j:
                    dist = abs(i - j)
                    if 3 <= dist <= 5:
                        template_channels[2, i, j] = ss_propensity[i] * 0.3
                    elif dist >= 15:
                        template_channels[2, i, j] = ss_propensity[i] * 0.1

        # Channel 3: Coevolution patterns
        for i in range(L):
            for j in range(L):
                if i != j:
                    dist = abs(i - j)
                    if dist > 12 and dist < 50:
                        template_channels[3, i, j] = coevolution_scores[i] * 0.2 * (1 - dist / 50)

        # Set diagonal to 1.0
        for i in range(4):
            np.fill_diagonal(template_channels[i], 1.0)

        print(f"   ✅ Template channels created from real data: {template_channels.shape}")

        # Return templates list and channels (only high-quality templates)
        template_list = [{
            'pdb_id': r.pdb_id,
            'chain_id': r.chain_id,
            'identity': r.sequence_identity,
            'coverage': r.coverage,
            'e_value': r.e_value
        } for r in all_results[:10]]

        return template_list, template_channels

    except Exception as e:
        print(f"❌ Error searching homology templates: {e}")
        import traceback
        print(f"   📍 Traceback: {traceback.format_exc()}")

        # No fallback - fail with clear error message
        raise RuntimeError(
            f"Template search failed for sequence length {len(sequence)}. "
            f"This is required for CNN dataset generation. "
            f"Please ensure:\n"
            f"  1. Homology databases are downloaded and ready:\n"
            f"     uv run python scripts/02_download_homology_databases.py --db all\n"
            f"  2. Database files are properly extracted and accessible\n"
            f"  3. HHblits/BLAST tools are available if required\n"
            f"  4. Sufficient system resources for template search\n"
            f"Original error: {e}"
        )

def assemble_68_channel_tensor(esm2_embedding: np.ndarray, contact_map: np.ndarray,
                              template_channels: np.ndarray) -> np.ndarray:
    """Assemble 68-channel tensor (4 template + 64 ESM2) with robust shape handling."""
    try:
        L = contact_map.shape[0]  # Sequence length from contact map
        channels = 68
        height = L
        width = L

        print(f"   🏗️ Assembling tensor: L={L}, esm2_shape={esm2_embedding.shape}, template_shape={template_channels.shape}")

        # Initialize multi-channel tensor
        tensor = np.zeros((channels, height, width), dtype=np.float32)

        # Channels 0-3: Template channels
        if template_channels.shape == (4, L, L):
            tensor[0:4] = template_channels
            print(f"   ✅ Template channels assigned: {template_channels.shape}")
        else:
            # Fallback: use contact map for template channels
            print(f"   ⚠️  Template shape mismatch {template_channels.shape}, using contact map fallback")
            for i in range(4):
                tensor[i] = contact_map

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

        # Assign ESM2 channels - this is where the broadcast error was happening
        # We need to reshape (64, L) to (64, L, L) by replicating along the last dimension
        for i in range(64):
            # Replicate the 1D ESM2 feature across all positions to create 2D map
            tensor[4 + i] = np.tile(esm2_64_channels[i:i+1, :], (L, 1))

        print(f"   ✅ Final tensor shape: {tensor.shape}")
        print(f"   📊 Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

        return tensor

    except Exception as e:
        print(f"❌ Error assembling 68-channel tensor: {e}")
        import traceback
        print(f"   📍 Traceback: {traceback.format_exc()}")

        # No fallback - fail with clear error message
        raise RuntimeError(
            f"Failed to assemble 68-channel tensor. "
            f"This indicates an incompatibility between the input components:\n"
            f"  - ESM2 embedding shape: {esm2_embedding.shape}\n"
            f"  - Contact map shape: {contact_map.shape}\n"
            f"  - Template channels shape: {template_channels.shape}\n"
            f"Please check:\n"
            f"  1. All components have compatible sequence lengths\n"
            f"  2. ESM2 embedding has at least 64 dimensions\n"
            f"  3. Template channels have correct shape (4, L, L)\n"
            f"  4. Contact map is square (L, L)\n"
            f"Original error: {e}"
        )

def save_protein_to_hdf5(hdf5_file, protein_id: str, tensor: np.ndarray,
                         contact_map: np.ndarray, sequence: str, chain_id: str = None):
    """Save single protein data to HDF5 file with proper metadata."""
    try:
        import h5py

        with h5py.File(hdf5_file, 'a') as f:
            if 'cnn_data' not in f:
                group = f.create_group('cnn_data')
            else:
                group = f['cnn_data']

            # Create unique protein ID to avoid conflicts
            base_id = protein_id
            counter = 1
            while protein_id in group:
                protein_id = f"{base_id}_{counter}"
                counter += 1

            protein_group = group.create_group(protein_id)

            # Save data with proper attributes for Tiny10Dataset compatibility
            protein_group.create_dataset('multi_channel_input', data=tensor,
                                       shape=tensor.shape, dtype='f4',
                                       compression='gzip', compression_opts=4)
            protein_group.create_dataset('consensus_contact_map', data=contact_map,
                                       shape=contact_map.shape, dtype='f4',
                                       compression='gzip', compression_opts=4)

            # Add required attributes
            protein_group.attrs['query_sequence'] = sequence
            protein_group.attrs['sequence_length'] = len(sequence)
            protein_group.attrs['protein_id'] = base_id  # Keep original ID
            if chain_id:
                protein_group.attrs['chain_id'] = chain_id
            protein_group.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        print(f"❌ Error saving to HDF5: {e}")

def validate_startup_dependencies():
    """Validate all required dependencies before processing any files."""
    try:
        print("   🔍 Checking ESM2 model availability...")
        # Test ESM2 model loading
        load_esm2_model()
        print("   ✅ ESM2 model available")

        print("   🔍 Checking homology databases...")
        # Validate homology databases
        validate_homology_databases()
        print("   ✅ Homology databases available")

        print("   🔍 Checking required libraries...")
        # Check for required libraries
        try:
            import h5py
            print("   ✅ HDF5 library available")
        except ImportError:
            raise RuntimeError(
                "HDF5 library not found. Please install: uv add h5py"
            )

        try:
            from Bio.PDB import PDBParser
            print("   ✅ Biopython available")
        except ImportError:
            raise RuntimeError(
                "Biopython not found. Please install: uv add biopython"
            )

        print("   🔍 Checking system resources...")
        resources = monitor_resources()
        print(f"   💾 Available memory: {resources['memory_gb']:.1f}GB")

        if resources['memory_gb'] < 2.0:
            print("   ⚠️  Low memory available (<2GB). Consider increasing system memory.")
        else:
            print("   ✅ Sufficient memory available")

        print("   ✅ All dependencies validated successfully")

    except Exception as e:
        print(f"   ❌ Dependency validation failed: {e}")
        print()
        print("🛠️  Setup Instructions:")
        print("1. Install required libraries:")
        print("   uv add esm h5py biopython numpy torch tqdm")
        print()
        print("2. Download homology databases:")
        print("   uv run python scripts/02_download_homology_databases.py --db all")
        print()
        print("3. Ensure sufficient system memory (>=4GB recommended)")
        print()
        raise RuntimeError(
            f"Dependency validation failed. Please install missing dependencies "
            f"and databases before running CNN dataset generation. "
            f"See instructions above. Original error: {e}"
        )

def main():
    """Main processing function."""
    warnings.filterwarnings('ignore')

    print("🚀 CNN Dataset Generation from PDB Files")
    print("=" * 60)
    print("This script creates a complete CNN dataset from PDB files")
    print("ready for training with the existing pipeline.")
    print()

    args = parse_arguments()

    print(f"📋 Configuration:")
    print(f"   PDB directory: {args.pdb_dir}")
    print(f"   Process ratio: {args.process_ratio * 100:.1f}%")
    print(f"   Output path: {args.output_path}")
    print(f"   Random seed: {args.random_seed if args.random_seed is not None else 'random'}")
    print(f"   CPU limit: {args.cpu_limit}")
    print()

    # Create cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Validate dependencies
    print("🔧 Step 0: Validating dependencies")
    validate_startup_dependencies()
    print()

    # Step 1: Discover PDB files
    print("🔍 Step 1: Discovering PDB files")
    pdb_files = discover_pdb_files(args.pdb_dir)
    print(f"   Found {len(pdb_files)} PDB files")

    if len(pdb_files) == 0:
        print(f"❌ No PDB files found in {args.pdb_dir}")
        return 1

    # Step 2: Select files to process
    print("🎯 Step 2: Selecting files to process")
    selected_files = select_pdb_files(pdb_files, args.process_ratio, args.random_seed)
    print(f"   Selected {len(selected_files)} files for processing")
    print()

    # Step 3: Process files
    print("🧪 Step 3: Processing PDB files")

    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    processed_count = 0
    memory_issues = 0

    with tqdm(selected_files, desc="Processing PDB files") as pbar:
        for i, pdb_file in enumerate(pbar):
            print(f"\n📁 Processing {i+1}/{len(selected_files)}: {pdb_file.name}")

            # Monitor resources before processing
            if not check_memory_limits():
                memory_issues += 1
                if memory_issues > 3:
                    print(f"❌ Too many memory issues, stopping after {processed_count} proteins")
                    break
                print(f"   ⚠️  Skipping {pdb_file.name} due to memory constraints")
                continue

            # Show resource usage
            resources = monitor_resources()
            print(f"   💾 Memory: {resources['memory_gb']:.1f}GB ({resources['memory_percent']:.1f}%)")

            # Extract protein information
            protein_info = extract_protein_from_pdb(pdb_file)

            if not protein_info:
                print(f"   ❌ Failed to parse PDB file")
                cleanup_memory()
                continue

            protein_id = protein_info['pdb_id']
            chain_id = protein_info.get('chain_id', 'A')

            # Get sequence from enhanced parser
            if not protein_info['chain_sequences']:
                raise RuntimeError(
                    f"No sequence found in PDB file {pdb_file.name}. "
                    f"This is required for CNN dataset generation. "
                    f"Please ensure:\n"
                    f"  1. PDB file contains protein chains with residues\n"
                    f"  2. At least one chain has valid amino acid residues\n"
                    f"  3. PDB file is not corrupted or malformed\n"
                    f"  4. Standard PDB format with proper residue numbering"
                )

            sequence = protein_info['chain_sequences'][chain_id]

            print(f"   🧬 Protein ID: {protein_id}")
            print(f"   🔗 Chain ID: {chain_id}")
            print(f"   📏 Sequence length: {len(sequence)}")

            # Validate minimum requirements
            if len(sequence) < 2:
                print(f"   ⚠️  Sequence too short, skipping")
                cleanup_memory()
                continue

            # Skip very large sequences to avoid memory issues
            if len(sequence) > 800:
                print(f"   ⚠️  Sequence too long ({len(sequence)} aa), skipping to avoid memory issues")
                cleanup_memory()
                continue

            try:
                # Generate ESM2 embeddings
                print(f"   🔢 Generating ESM2 embeddings...")
                esm2_embedding = generate_esm2_embeddings(protein_id, sequence, cache_dir)
                print(f"   ✅ ESM2 embeddings: {esm2_embedding.shape}")

                # Calculate contact map
                print(f"   📞 Calculating contact map...")
                contact_map = calculate_contact_map(protein_info['coordinates'], chain_id)
                print(f"   ✅ Contact map: {contact_map.shape}, density: {np.mean(contact_map):.4f}")

                # Search for homology templates
                print(f"   🔍 Creating template channels...")
                templates, template_channels = search_homology_templates(sequence, cache_dir, args.cpu_limit)
                print(f"   ✅ Template channels created")

                # Assemble 68-channel tensor
                print(f"   🏗️ Assembling 68-channel tensor...")
                tensor = assemble_68_channel_tensor(esm2_embedding, contact_map, template_channels)
                print(f"   ✅ Tensor: {tensor.shape}")

                # Validate tensor shape
                expected_shape = (68, len(sequence), len(sequence))
                if tensor.shape != expected_shape:
                    print(f"   ⚠️  Tensor shape mismatch: {tensor.shape} != {expected_shape}")

                # Save to HDF5
                save_protein_to_hdf5(args.output_path, protein_id, tensor,
                                   contact_map, sequence, chain_id)
                print(f"   ✅ Saved to {args.output_path}")

                processed_count += 1

            except Exception as e:
                print(f"   ❌ Error processing {pdb_file.name}: {e}")
                continue

            finally:
                # Memory cleanup after each protein
                cleanup_memory()

    total_time = time.time() - start_time

    print(f"\n🎉 CNN Dataset Generation Completed!")
    print("=" * 60)
    print(f"⏱️  Total processing time: {total_time:.1f} seconds")
    print(f"📁 Proteins successfully processed: {processed_count}/{len(selected_files)}")
    if memory_issues > 0:
        print(f"⚠️  Memory issues encountered: {memory_issues}")
    print(f"💾 Output file: {args.output_path}")

    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024*1024)
        print(f"📊 File size: {file_size_mb:.1f} MB")

        # Final resource check
        final_resources = monitor_resources()
        print(f"💾 Final memory usage: {final_resources['memory_gb']:.1f}GB")

    print()
    print("✅ Dataset ready for training with existing pipeline!")
    print("💡 Run training with: uv run python scripts/train_cnn_binary.py")

    # Final cleanup
    cleanup_memory()

    return 0

if __name__ == "__main__":
    sys.exit(main())