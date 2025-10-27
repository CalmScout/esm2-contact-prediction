#!/usr/bin/env python3
"""
ESM-2 Embeddings Computation Script

This script computes ESM-2 embeddings for protein contact prediction datasets.
It integrates with existing HDF5 datasets to add embeddings alongside contact maps.

Usage:
    python scripts/compute_esm2_embeddings.py --mode test --verbose
    python scripts/compute_esm2_embeddings.py --mode all --batch-size 6
"""

import argparse
import time
import sys
import os
import signal
from pathlib import Path
import warnings
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Suppress warnings
warnings.filterwarnings('ignore')

# Process locking
PID_FILE = Path(__file__).parent / ".esm2_embeddings.lock"

# ESM-2 Model specifications and requirements
MODEL_REQUIREMENTS = {
    'auto': {
        'name': 'Auto-selection based on available memory',
        'min_memory_gb': 0,
        'embedding_dim': 0,
        'layers': 0,
        'parameters': 'Auto',
        'description': 'Automatically selects the best model for your hardware'
    },
    'esm2_t6_8M_UR50D': {
        'name': 'ESM-2 Small (8M parameters)',
        'min_memory_gb': 2,
        'embedding_dim': 320,
        'layers': 6,
        'parameters': '8M',
        'description': 'Fastest model, good for quick tests or resource-constrained systems'
    },
    'esm2_t12_35M_UR50D': {
        'name': 'ESM-2 Base (35M parameters)',
        'min_memory_gb': 4,
        'embedding_dim': 480,
        'layers': 12,
        'parameters': '35M',
        'description': 'Good balance of speed and performance, works on most GPUs'
    },
    'esm2_t30_150M_UR50D': {
        'name': 'ESM-2 Medium (150M parameters)',
        'min_memory_gb': 6,
        'embedding_dim': 640,
        'layers': 30,
        'parameters': '150M',
        'description': 'Better performance for medium-sized proteins'
    },
    'esm2_t33_650M_UR50D': {
        'name': 'ESM-2 Large (650M parameters)',
        'min_memory_gb': 8,
        'embedding_dim': 1280,
        'layers': 33,
        'parameters': '650M',
        'description': 'Recommended for most applications, excellent performance'
    },
    'esm2_t36_3B_UR50D': {
        'name': 'ESM-2 X-Large (3B parameters)',
        'min_memory_gb': 16,
        'embedding_dim': 2560,
        'layers': 36,
        'parameters': '3B',
        'description': 'High performance, requires substantial GPU memory'
    },
    'esm2_t48_15B_UR50D': {
        'name': 'ESM-2 XX-Large (15B parameters)',
        'min_memory_gb': 32,
        'embedding_dim': 5120,
        'layers': 48,
        'parameters': '15B',
        'description': 'Maximum performance, requires high-end GPU with lots of memory'
    }
}

def check_existing_process():
    """Check if another instance is already running."""
    if PID_FILE.exists():
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())

            # Check if process is still running
            try:
                os.kill(pid, 0)  # Send signal 0 to check if process exists
                return pid  # Process is still running
            except OSError:
                # Process not found, remove stale PID file
                PID_FILE.unlink(missing_ok=True)
                return None
        except (ValueError, IOError):
            # Invalid PID file, remove it
            PID_FILE.unlink(missing_ok=True)
            return None
    return None

def create_lock_file():
    """Create PID lock file."""
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        return True
    except IOError as e:
        print(f"❌ Failed to create lock file: {e}")
        return False

def remove_lock_file():
    """Remove PID lock file."""
    PID_FILE.unlink(missing_ok=True)

def signal_handler(signum, frame):
    """Handle signals for cleanup."""
    print(f"\n🛑 Received signal {signum}. Cleaning up...")
    remove_lock_file()
    sys.exit(1)

def check_gpu_resources():
    """Check GPU resources and warn if in use."""
    try:
        import torch
        if torch.cuda.is_available():
            # Check GPU memory usage
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9

            if allocated > 1.0:  # More than 1GB in use
                print(f"⚠️  GPU memory already in use: {allocated:.1f}GB / {total:.1f}GB")
                print("   Another process might be using the GPU.")
                print("   Consider waiting for it to complete or using --mode with smaller batches.")
                return False
            else:
                print(f"✅ GPU memory available: {allocated:.1f}GB / {total:.1f}GB")
                return True
        return True
    except Exception as e:
        print(f"⚠️  Could not check GPU resources: {e}")
        return True

def validate_model_selection(model_name: str, verbose: bool = True):
    """Validate model selection against available resources."""
    if model_name not in MODEL_REQUIREMENTS:
        print(f"❌ Invalid model: {model_name}")
        print(f"   Available models: {', '.join(MODEL_REQUIREMENTS.keys())}")
        return False

    if model_name == 'auto':
        return True  # Auto selection is always valid

    model_info = MODEL_REQUIREMENTS[model_name]

    # Check GPU memory requirements
    import torch
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        required_memory = model_info['min_memory_gb']

        if total_memory < required_memory:
            if verbose:
                print(f"❌ Insufficient GPU memory for {model_info['name']}")
                print(f"   Required: {required_memory}GB, Available: {total_memory:.1f}GB")
                print(f"   Consider using a smaller model or CPU with --model auto")
            return False
        else:
            if verbose:
                print(f"✅ Sufficient GPU memory for {model_info['name']}")
                print(f"   Required: {required_memory}GB, Available: {total_memory:.1f}GB")
    else:
        if model_info['min_memory_gb'] > 4:
            if verbose:
                print(f"⚠️  {model_info['name']} recommended for GPU")
                print(f"   Running on CPU may be very slow")
                response = input("Continue anyway? (y/N): ")
                return response.lower() == 'y'

    return True

def get_auto_model_selection():
    """Get automatic model selection based on available resources."""
    import torch

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        if total_memory >= 32:
            return 'esm2_t48_15B_UR50D'
        elif total_memory >= 16:
            return 'esm2_t36_3B_UR50D'
        elif total_memory >= 8:
            return 'esm2_t33_650M_UR50D'
        elif total_memory >= 6:
            return 'esm2_t30_150M_UR50D'
        elif total_memory >= 4:
            return 'esm2_t12_35M_UR50D'
        else:
            return 'esm2_t6_8M_UR50D'
    else:
        return 'esm2_t12_35M_UR50D'  # CPU fallback

def show_model_info(model_name: str):
    """Display detailed information about a model."""
    if model_name not in MODEL_REQUIREMENTS:
        print(f"❌ Unknown model: {model_name}")
        return

    model_info = MODEL_REQUIREMENTS[model_name]

    print(f"\n🤖 Model Information: {model_info['name']}")
    print("=" * 60)
    print(f"📊 Parameters: {model_info['parameters']}")
    print(f"🧠 Embedding Dimensions: {model_info['embedding_dim']}")
    print(f"📚 Layers: {model_info['layers']}")
    print(f"💾 Minimum GPU Memory: {model_info['min_memory_gb']}GB")
    print(f"📝 Description: {model_info['description']}")

    if model_name != 'auto':
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🖥️  Available GPU Memory: {total_memory:.1f}GB")
            if total_memory >= model_info['min_memory_gb']:
                print("✅ This model is compatible with your GPU")
            else:
                print("❌ This model requires more GPU memory than available")
        else:
            print("⚠️  Running on CPU - computation may be slow")

    print("=" * 60)

def verify_embeddings_files(output_paths, verbose: bool = True):
    """Verify existing embedding files."""
    from notebook_utils.esm2_embeddings import get_hdf5_embeddings_info

    all_valid = True

    for dataset_name, output_path in output_paths.items():
        if output_path.exists():
            if verbose:
                print(f"\n📁 Verifying {dataset_name} embeddings...")

            try:
                info = get_hdf5_embeddings_info(output_path)

                if 'error' in info:
                    print(f"❌ Error reading {dataset_name} embeddings: {info['error']}")
                    all_valid = False
                    continue

                if verbose:
                    print(f"✅ {dataset_name} embeddings are valid")
                    print(f"   Total embeddings: {info.get('total_embeddings', 0):,}")
                    print(f"   Model: {info.get('model_name', 'Unknown')}")
                    print(f"   Embedding dimensions: {info.get('embedding_dim', 'Unknown')}")
                    print(f"   File size: {info.get('file_size_mb', 0):.1f} MB")
                    print(f"   Compression: {info.get('compression_level', 'Unknown')}")
                    print(f"   Extraction date: {info.get('extraction_date', 'Unknown')}")

                    # Check source dataset
                    source_path = output_path.parent / f"{dataset_name}_contacts.h5"
                    if source_path.exists():
                        from esm2_contact.dataset.utils import ContactDataset
                        dataset = ContactDataset(str(source_path))
                        total_chains = len(dataset)
                        embeddings_count = info.get('total_embeddings', 0)

                        if embeddings_count == total_chains:
                            print(f"   ✅ Complete: All {total_chains:,} chains have embeddings")
                        else:
                            missing = total_chains - embeddings_count
                            print(f"   ⚠️  Incomplete: {missing:,} chains missing embeddings")
                            all_valid = False
                    else:
                        print(f"   ⚠️  Source dataset not found: {source_path}")

            except Exception as e:
                print(f"❌ Error verifying {dataset_name} embeddings: {e}")
                all_valid = False
        else:
            if verbose:
                print(f"❌ {dataset_name} embeddings file not found: {output_path}")
            all_valid = False

    return all_valid

def load_model_and_components(model_name: str = 'auto'):
    """Load ESM-2 model and required components."""
    import torch
    import esm

    print("🔧 Loading ESM-2 model and components...")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Handle model selection
    if model_name == 'auto':
        model_name = get_auto_model_selection()
        print(f"🤖 Auto-selected model: {model_name}")
    else:
        print(f"🎯 Using specified model: {model_name}")

    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()

    # Memory optimization
    if device == 'cuda':
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("🔧 Gradient checkpointing enabled")
        torch.cuda.empty_cache()
        print("🗑️  GPU cache cleared")

    # Model info
    model_info = {
        'model_name': model_name,
        'embedding_dim': model.embed_dim,
        'layers': model.num_layers,
        'params': f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M"
    }

    print(f"✅ Model loaded successfully!")
    print(f"📊 Embedding dimension: {model_info['embedding_dim']}")
    print(f"🧠 Layers: {model_info['layers']}")
    print(f"📈 Parameters: {model_info['params']}")

    return model, batch_converter, model_info, device

def import_required_functions():
    """Import required functions from notebook utilities."""
    try:
        # Import from notebook utilities
        from notebook_utils.esm2_embeddings import (
            compute_esm2_embeddings,
            store_embeddings_hdf5,
            process_hdf5_embeddings,
            get_hdf5_embeddings_info,
            prepare_sequences_for_esm
        )
        return compute_esm2_embeddings, store_embeddings_hdf5, process_hdf5_embeddings, get_hdf5_embeddings_info
    except ImportError:
        print("❌ Cannot import from notebook_utils. Using local implementations...")
        # Fallback to local implementations if available
        from esm2_contact.dataset.utils import ContactDataset
        return None, None, None, ContactDataset

def get_dataset_info(data_dir, dataset_name):
    """Get information about a dataset."""
    from esm2_contact.dataset.utils import ContactDataset

    dataset_path = data_dir / f"{dataset_name}_contacts.h5"
    if not dataset_path.exists():
        return None

    try:
        dataset = ContactDataset(str(dataset_path))
        return {
            'path': dataset_path,
            'total_chains': len(dataset),
            'output_path': data_dir / f"{dataset_name}_with_embeddings.h5"
        }
    except Exception as e:
        print(f"❌ Error reading {dataset_name} dataset: {e}")
        return None

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.1f} MB"
    else:
        return f"{size_bytes/1024**3:.1f} GB"

def process_single_dataset(mode, data_dir, batch_size, compression_level, verbose, sample_size=None, global_progress_callback=None, model_name='auto'):
    """Process a single dataset with improved progress tracking."""
    from esm2_contact.dataset.utils import ContactDataset

    if mode == 'train':
        dataset_name = 'train'
    elif mode == 'test':
        dataset_name = 'test'
    elif mode == 'test_sample':
        dataset_name = 'test'
    else:
        raise ValueError(f"Invalid single dataset mode: {mode}")

    input_hdf5 = data_dir / f"{dataset_name}_contacts.h5"
    output_hdf5 = data_dir / f"{dataset_name}_with_embeddings.h5"

    if not input_hdf5.exists():
        print(f"❌ Input file not found: {input_hdf5}")
        return False, 0

    # Load dataset to get total chains
    dataset = ContactDataset(str(input_hdf5))
    total_chains = len(dataset)

    # Calculate sample size if specified
    if sample_size and sample_size < total_chains:
        actual_chains = sample_size
        print(f"📊 Processing {actual_chains:,} chains from {dataset_name} dataset (sample)")
    else:
        actual_chains = total_chains
        if verbose:
            print(f"📊 Processing {actual_chains:,} chains from {dataset_name} dataset")
        else:
            print(f"📊 Processing {dataset_name} dataset: {actual_chains:,} chains")

    # Set resume for non-sample modes
    resume = mode != 'test_sample'

    # Load model and components (only show details in verbose mode)
    import torch
    import esm
    import gc

    if verbose:
        model, batch_converter, model_info, device = load_model_and_components(model_name)
    else:
        # Silent model loading using specified model
        model, batch_converter, model_info, device = load_model_and_components(model_name)

    # Import required functions
    compute_esm2_embeddings, store_embeddings_hdf5, process_hdf5_embeddings, get_hdf5_embeddings_info = import_required_functions()

    if process_hdf5_embeddings is None:
        print("❌ Failed to import required functions")
        return False, 0

    # Process embeddings
    start_time = time.time()

    try:
        process_hdf5_embeddings(
            input_hdf5=input_hdf5,
            output_hdf5=output_hdf5,
            model=model,
            batch_converter=batch_converter,
            model_info=model_info,
            batch_size=batch_size,
            sample_size=sample_size,
            resume=resume,
            compression_level=compression_level,
            verbose=verbose,
            global_progress_callback=global_progress_callback
        )

        # Calculate processing time
        elapsed_time = time.time() - start_time

        # Validate output
        if output_hdf5.exists():
            file_size = output_hdf5.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Get embeddings info
            embeddings_info = get_hdf5_embeddings_info(output_hdf5)
            processed_embeddings = embeddings_info.get('total_embeddings', 0)

            if verbose:
                print(f"\n📁 {dataset_name} dataset completed:")
                print(f"  Chains processed: {processed_embeddings:,}")
                print(f"  File size: {format_file_size(file_size)}")
                print(f"  Processing time: {elapsed_time/60:.1f} minutes")
                print(f"  Processing speed: {processed_embeddings/(elapsed_time/60):.1f} chains/min")
                if 'model_name' in embeddings_info:
                    print(f"  Model: {embeddings_info['model_name']}")
                    print(f"  Embedding dim: {embeddings_info.get('embedding_dim', 'Unknown')}")
            else:
                compression_ratio = file_size / (processed_embeddings * 480 * 4) if processed_embeddings > 0 else 1
                print(f"✅ {dataset_name}: {processed_embeddings:,} chains → {format_file_size(file_size)}")

            return True, processed_embeddings
        else:
            print(f"❌ Output file not created: {output_hdf5}")
            return False, 0

    except KeyboardInterrupt:
        print(f"\n🛑 {dataset_name} dataset processing interrupted by user")
        raise
    except Exception as e:
        error_msg = str(e)
        if "Unable to synchronously open file" in error_msg or "unable to lock file" in error_msg:
            print(f"❌ File locking error processing {dataset_name} dataset:")
            print(f"   This usually happens when multiple processes try to access the same files.")
            print(f"   Details: {error_msg}")
            print(f"   Solution: Wait for other processes to complete, or run: rm -f scripts/.esm2_embeddings.lock")
        elif "CUDA out of memory" in error_msg:
            print(f"❌ GPU memory error processing {dataset_name} dataset:")
            print(f"   Details: {error_msg}")
            print(f"   Solution: Try smaller batch size with --batch-size 4 or restart the process")
        elif "torch" in error_msg.lower() or "esm" in error_msg.lower():
            print(f"❌ Model error processing {dataset_name} dataset:")
            print(f"   Details: {error_msg}")
            print(f"   Solution: Check PyTorch and ESM installations")
        else:
            print(f"❌ Error processing {dataset_name} dataset: {error_msg}")
            print(f"   This might be a data corruption or I/O issue")

        if verbose:
            import traceback
            traceback.print_exc()
        return False, 0
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_embeddings(mode, sample_size=None, batch_size=6, compression_level=6, verbose=False, model_name='auto', info_only=False, verify_only=False):
    """
    Process ESM-2 embeddings for specified dataset.

    Args:
        mode: 'train', 'test', 'test_sample', or 'all'
        sample_size: Number of chains to process (None for all)
        batch_size: Batch size for embedding computation
        compression_level: HDF5 compression level
        verbose: Show detailed progress information
    """
    import time
    from tqdm import tqdm

    # Handle info-only mode
    if info_only:
        show_model_info(model_name)
        return True

    print("🧬 ESM-2 Embeddings Computation")
    if verbose:
        print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Compression level: {compression_level}")
    if mode == "test_sample":
        print(f"Sample size: {sample_size}")
    if not verbose:
        print("🔧 Use --verbose for detailed progress information")
    if verbose:
        print("=" * 60)

    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "processed"

    # Validate model selection
    if not validate_model_selection(model_name, verbose):
        return False

    # Handle verify-only mode
    if verify_only:
        print("🔍 Verification mode - checking existing embeddings...")
        output_paths = {}

        if mode == 'all':
            output_paths['train'] = data_dir / "train_with_embeddings.h5"
            output_paths['test'] = data_dir / "test_with_embeddings.h5"
        elif mode == 'train':
            output_paths['train'] = data_dir / "train_with_embeddings.h5"
        elif mode == 'test':
            output_paths['test'] = data_dir / "test_with_embeddings.h5"
        elif mode == 'test_sample':
            output_paths['test'] = data_dir / "test_with_embeddings.h5"

        all_valid = verify_embeddings_files(output_paths, verbose=True)

        if all_valid:
            print("\n✅ All embedding files are valid and complete!")
        else:
            print("\n❌ Some embedding files have issues. Check the output above for details.")

        return all_valid

    # Handle different modes
    if mode == 'all':
        # Process both train and test datasets
        if verbose:
            print("📁 Processing all datasets (train + test)")
        else:
            print("📁 Processing all datasets")

        # Calculate total chains across all datasets for global progress
        train_info = get_dataset_info(data_dir, 'train')
        test_info = get_dataset_info(data_dir, 'test')

        train_chains = train_info['total_chains'] if train_info else 0
        test_chains = test_info['total_chains'] if test_info else 0
        total_chains = train_chains + test_chains

        # Create global progress bar
        global_pbar = tqdm(total=total_chains, desc="Processing embeddings",
                          unit="chains", disable=verbose,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        total_processed = 0
        all_success = True
        start_time = time.time()

        # Progress callback function
        def update_global_progress(chains_processed):
            global_pbar.update(chains_processed)

        # Process training dataset
        if verbose:
            print("\n" + "="*40)
            print("TRAINING DATASET")
            print("="*40)

        train_success, train_processed = process_single_dataset(
            'train', data_dir, batch_size, compression_level, verbose, sample_size=None, global_progress_callback=update_global_progress, model_name=model_name
        )

        if train_success:
            total_processed += train_processed
        else:
            all_success = False

        # Process test dataset
        if verbose:
            print("\n" + "="*40)
            print("TEST DATASET")
            print("="*40)

        test_success, test_processed = process_single_dataset(
            'test', data_dir, batch_size, compression_level, verbose, sample_size=None, global_progress_callback=update_global_progress, model_name=model_name
        )

        if test_success:
            total_processed += test_processed
        else:
            all_success = False

        # Close global progress bar
        global_pbar.close()

        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n🎉 All embeddings completed!")
        print(f"   Total chains processed: {total_processed:,}")
        print(f"   Total processing time: {elapsed_time/60:.1f} minutes")
        if total_processed > 0:
            print(f"   Average speed: {total_processed/(elapsed_time/60):.1f} chains/min")

        return all_success

    elif mode in ['train', 'test', 'test_sample']:
        # Process single dataset
        success, processed = process_single_dataset(
            mode, data_dir, batch_size, compression_level, verbose, sample_size, model_name=model_name
        )

        if success:
            if verbose:
                print(f"\n✅ Successfully processed {mode} dataset!")
            return True
        else:
            return False

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'train', 'test', 'test_sample', or 'all'")

def main():
    """Main function."""
    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check for existing process
    existing_pid = check_existing_process()
    if existing_pid:
        print("❌ Another ESM-2 embeddings computation is already running!")
        print(f"   Process ID: {existing_pid}")
        print("   Please wait for it to complete or terminate it manually.")
        print()
        print("   To check if it's still active:")
        print(f"     ps -p {existing_pid}")
        print()
        print("   To terminate it (if you're sure it's safe):")
        print(f"     kill {existing_pid}")
        sys.exit(1)

    # Create lock file
    if not create_lock_file():
        print("❌ Failed to create lock file. Another instance might be running.")
        sys.exit(1)

    try:
        parser = argparse.ArgumentParser(
            description="Compute ESM-2 embeddings for protein datasets",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process all datasets silently (recommended)
  python scripts/compute_esm2_embeddings.py --mode all --batch-size 6

  # Use specific model
  python scripts/compute_esm2_embeddings.py --mode all --model esm2_t33_650M_UR50D

  # Show model information
  python scripts/compute_esm2_embeddings.py --model esm2_t36_3B_UR50D --info

  # Verify existing embeddings
  python scripts/compute_esm2_embeddings.py --mode all --verify

  # Process with detailed logging
  python scripts/compute_esm2_embeddings.py --mode all --verbose

  # Process individual datasets
  python scripts/compute_esm2_embeddings.py --mode train --batch-size 8
  python scripts/compute_esm2_embeddings.py --mode test --batch-size 6

  # Test with small sample
  python scripts/compute_esm2_embeddings.py --mode test_sample --sample-size 50 --verbose

  # Force start (bypass lock file - use with caution)
  rm -f scripts/.esm2_embeddings.lock
  python scripts/compute_esm2_embeddings.py --mode all

Model Options:
  auto                    - Auto-select best model for your hardware (default)
  esm2_t6_8M_UR50D      - 8M parameters, 320 dim, fastest, minimal GPU memory
  esm2_t12_35M_UR50D    - 35M parameters, 480 dim, good balance
  esm2_t30_150M_UR50D   - 150M parameters, 640 dim, better performance
  esm2_t33_650M_UR50D   - 650M parameters, 1280 dim, recommended
  esm2_t36_3B_UR50D     - 3B parameters, 2560 dim, high performance
  esm2_t48_15B_UR50D    - 15B parameters, 5120 dim, maximum performance
            """
        )

        parser.add_argument("--mode",
                            choices=["train", "test", "test_sample", "all"],
                            default="test_sample",
                            help="Dataset mode: train, test, test_sample, or all (default: test_sample)")
        parser.add_argument("--sample-size", type=int, default=100,
                            help="Number of chains to process for test_sample mode (default: 100)")
        parser.add_argument("--batch-size", type=int, default=6,
                            help="Batch size for embedding computation (default: 6)")
        parser.add_argument("--compression", type=int, default=6,
                            help="HDF5 compression level (1-9, default: 6)")
        parser.add_argument("--model",
                            choices=list(MODEL_REQUIREMENTS.keys()),
                            default="auto",
                            help="ESM-2 model to use (default: auto)")
        parser.add_argument("--info", action="store_true",
                            help="Show model information and exit")
        parser.add_argument("--verify", action="store_true",
                            help="Verify existing embeddings without computing")
        parser.add_argument("--verbose", action="store_true",
                            help="Show detailed progress information (default: silent)")

        args = parser.parse_args()

        # Check GPU resources
        if not check_gpu_resources():
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborting due to GPU resource conflicts.")
                sys.exit(1)

        # Process based on mode
        if args.mode == "test_sample":
            success = process_embeddings(
                mode="test_sample",
                sample_size=args.sample_size,
                batch_size=args.batch_size,
                compression_level=args.compression,
                verbose=args.verbose,
                model_name=args.model,
                info_only=args.info,
                verify_only=args.verify
            )
        elif args.mode == "train":
            success = process_embeddings(
                mode="train",
                sample_size=None,
                batch_size=args.batch_size,
                compression_level=args.compression,
                verbose=args.verbose,
                model_name=args.model,
                info_only=args.info,
                verify_only=args.verify
            )
        elif args.mode == "test":
            success = process_embeddings(
                mode="test",
                sample_size=None,
                batch_size=args.batch_size,
                compression_level=args.compression,
                verbose=args.verbose,
                model_name=args.model,
                info_only=args.info,
                verify_only=args.verify
            )
        elif args.mode == "all":
            success = process_embeddings(
                mode="all",
                sample_size=None,
                batch_size=args.batch_size,
                compression_level=args.compression,
                verbose=args.verbose,
                model_name=args.model,
                info_only=args.info,
                verify_only=args.verify
            )

        if success and not args.info and not args.verify:
            print("\n🎉 ESM-2 embeddings computation completed successfully!")
        elif success and args.verify:
            print("\n✅ Embedding verification completed!")
        elif not success:
            print("\n❌ Operation failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Cleaning up...")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Always remove lock file on exit
        remove_lock_file()

if __name__ == "__main__":
    main()