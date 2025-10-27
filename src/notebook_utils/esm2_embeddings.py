"""
ESM-2 Embeddings utilities extracted from notebook.

This module contains the core functions for computing and storing ESM-2 embeddings,
extracted from the Jupyter notebook for use in scripts and other modules.
"""

import time
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm.notebook import tqdm
import h5py
import esm
from esm2_contact.dataset.utils import ContactDataset

def retry_hdf5_operation(func: Callable, max_retries: int = 5, base_delay: float = 0.1, verbose: bool = True):
    """
    Retry wrapper for HDF5 operations with exponential backoff.

    Args:
        func: Function to retry (should return an HDF5 File object or handle)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (in seconds)
        verbose: Whether to log retry attempts

    Returns:
        Result of the function call

    Raises:
        Exception: If all retries are exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except (OSError, IOError) as e:
            if "Unable to synchronously open file" in str(e) or "unable to lock file" in str(e):
                if attempt == max_retries:
                    raise Exception(f"Failed to access HDF5 file after {max_retries} retries: {e}")

                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                if verbose:
                    print(f"⚠️  HDF5 file locked (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Re-raise non-locking errors immediately
                raise e
        except Exception as e:
            # Re-raise non-HDF5 errors immediately
            raise e

def compute_esm2_embeddings_batch(model, batch_converter, sequences_list: List[Tuple[str, str]],
                                  device: str = 'auto', batch_size: int = 8,
                                  layer: int = None, return_contacts: bool = False, verbose: bool = True):
    """
    Compute ESM-2 embeddings for a batch of sequences.

    Args:
        model: Loaded ESM-2 model
        batch_converter: ESM batch converter
        sequences_list: List of (identifier, sequence) tuples
        device: Device for computation ('auto', 'cuda', 'cpu')
        batch_size: Batch size for processing
        layer: Layer to extract embeddings from (None for final layer)
        return_contacts: Whether to return ESM attention contacts
        verbose: Whether to show detailed progress information

    Returns:
        Dictionary: {identifier: {'embedding': np.array, 'contacts': np.array (optional)}}
    """
    if device == 'auto':
        device = next(model.parameters()).device

    # Get model info for layer selection
    if layer is None:
        layer = model.num_layers  # Final layer

    if verbose:
        print(f"🔢 Computing embeddings from layer {layer}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🖥️  Device: {device}")

    results = {}
    total_sequences = len(sequences_list)

    # Process in batches
    progress_desc = "Computing embeddings" if verbose else None
    for i in tqdm(range(0, total_sequences, batch_size), desc=progress_desc, disable=not verbose):
        batch = sequences_list[i:i + batch_size]

        # Prepare batch
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            # Compute forward pass
            if return_contacts:
                outputs = model(
                    batch_tokens,
                    repr_layers=[layer],
                    return_contacts=True,
                    need_head_weights=False
                )
            else:
                outputs = model(
                    batch_tokens,
                    repr_layers=[layer],
                    return_contacts=False
                )

        # Extract embeddings and contacts
        embeddings = outputs["representations"][layer]  # Shape: (batch_size, seq_len + 2, embedding_dim)

        # Remove BOS and EOS tokens (first and last tokens)
        embeddings = embeddings[:, 1:-1, :]  # Shape: (batch_size, seq_len, embedding_dim)

        # Extract contacts if requested
        if return_contacts and "contacts" in outputs:
            contacts = outputs["contacts"]  # Shape: (batch_size, seq_len + 2, seq_len + 2)
            # Remove BOS/EOS from contacts
            contacts = contacts[:, 1:-1, 1:-1]  # Shape: (batch_size, seq_len, seq_len)
        else:
            contacts = None

        # Store results
        for j, (identifier, sequence) in enumerate(batch):
            # Move to CPU and convert to numpy
            embedding = embeddings[j].cpu().numpy()  # Shape: (seq_len, embedding_dim)

            results[identifier] = {
                'sequence': sequence,
                'embedding': embedding,
                'layer': layer,
                'embedding_dim': embedding.shape[-1]
            }

            if contacts is not None:
                results[identifier]['contacts'] = contacts[j].cpu().numpy()

        # Memory management
        del batch_tokens, embeddings, outputs
        if contacts is not None:
            del contacts

        if device == 'cuda':
            torch.cuda.empty_cache()

        # Small delay to prevent overheating
        time.sleep(0.1)

    if verbose:
        print(f"✅ Computed embeddings for {len(results)} sequences")
    return results

def prepare_sequences_for_esm(sequences_dict: Dict[str, str],
                              prefix: str = "") -> List[Tuple[str, str]]:
    """
    Prepare sequences for ESM-2 input format.

    Args:
        sequences_dict: Dictionary mapping chain_id to sequence
        prefix: Prefix for sequence identifiers

    Returns:
        List of (identifier, sequence) tuples for ESM batch converter
    """
    esm_sequences = []

    for chain_id, sequence in sequences_dict.items():
        # Create unique identifier
        identifier = f"{prefix}_{chain_id}" if prefix else f"protein_{chain_id}"

        # Clean sequence - replace invalid characters with X
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        clean_sequence = ''.join([aa if aa in valid_aa else 'X' for aa in sequence])

        esm_sequences.append((identifier, clean_sequence))

    return esm_sequences

def compute_esm2_embeddings(model, batch_converter, sequences_dict: Dict[str, str],
                           prefix: str = "", device: str = 'auto', batch_size: int = 8,
                           layer: int = None, return_contacts: bool = False, verbose: bool = True):
    """
    Compute ESM-2 embeddings for sequences dictionary.

    Args:
        model: Loaded ESM-2 model
        batch_converter: ESM batch converter
        sequences_dict: Dictionary mapping identifier to sequence
        prefix: Prefix for sequence identifiers
        device: Device for computation
        batch_size: Batch size for processing
        layer: Layer to extract embeddings from
        return_contacts: Whether to return attention contacts
        verbose: Whether to show detailed progress information

    Returns:
        Dictionary of embeddings with metadata
    """
    # Prepare sequences for ESM
    sequences_list = prepare_sequences_for_esm(sequences_dict, prefix=prefix)

    if not sequences_list:
        if verbose:
            print("❌ No sequences to process")
        return {}

    # Compute embeddings
    results = compute_esm2_embeddings_batch(
        model, batch_converter, sequences_list,
        device=device, batch_size=batch_size,
        layer=layer, return_contacts=return_contacts, verbose=verbose
    )

    return results

def store_embeddings_hdf5(embeddings_dict: Dict, hdf5_path: Path,
                          model_info: Dict, compression_level: int = 4, verbose: bool = True):
    """
    Store ESM-2 embeddings in HDF5 format, integrating with existing dataset structure.

    Args:
        embeddings_dict: Dictionary with embedding results
        hdf5_path: Path to HDF5 file (existing or new)
        model_info: Information about the ESM-2 model used
        compression_level: Compression level for embeddings (1-9)
        verbose: Whether to show detailed progress information
    """
    if verbose:
        print(f"💾 Storing embeddings to {hdf5_path}")
        print(f"📊 Total sequences: {len(embeddings_dict)}")
        print(f"🗜️  Compression level: {compression_level}")

    # Determine if we're creating new file or updating existing
    file_exists = hdf5_path.exists()

    def store_embeddings_operation():
        return h5py.File(hdf5_path, 'a')  # 'a' mode for read/write/create

    try:
        with retry_hdf5_operation(store_embeddings_operation, verbose=verbose) as f:
            # Create or get embeddings group
            if 'esm2_embeddings' not in f:
                embeddings_group = f.create_group('esm2_embeddings')

                # Store model metadata
                embeddings_group.attrs['model_name'] = model_info.get('model_name', 'unknown').encode('utf-8')
                embeddings_group.attrs['embedding_dim'] = model_info.get('embedding_dim', 0)
                embeddings_group.attrs['model_layers'] = model_info.get('layers', 0)
                embeddings_group.attrs['model_params'] = model_info.get('params', 'unknown').encode('utf-8')
                embeddings_group.attrs['extraction_date'] = time.strftime('%Y-%m-%d %H:%M:%S').encode('utf-8')
                embeddings_group.attrs['pytorch_version'] = torch.__version__.encode('utf-8')
                embeddings_group.attrs['esm_version'] = esm.__version__.encode('utf-8')

                if verbose:
                    print(f"📝 Created new embeddings group with model metadata")
            else:
                embeddings_group = f['esm2_embeddings']
                if verbose:
                    print(f"📝 Using existing embeddings group")

            # Store each embedding
            stored_count = 0
            skipped_count = 0

            progress_desc = "Storing embeddings" if verbose else None
            for identifier, data in tqdm(embeddings_dict.items(), desc=progress_desc, disable=not verbose):
                # Parse identifier (format: pdb_chain or prefix_chain)
                if '_' in identifier:
                    parts = identifier.split('_')
                    if len(parts) >= 2:
                        pdb_id = '_'.join(parts[:-1])
                        chain_id = parts[-1]
                    else:
                        pdb_id = parts[0]
                        chain_id = parts[1]
                else:
                    pdb_id = identifier
                    chain_id = 'A'  # default

                # Ensure proteins group exists
                if 'proteins' not in f:
                    proteins_group = f.create_group('proteins')
                    print(f"📁 Created proteins group")
                else:
                    proteins_group = f['proteins']

                # Ensure protein group exists
                if pdb_id not in proteins_group:
                    protein_group = proteins_group.create_group(pdb_id)
                else:
                    protein_group = proteins_group[pdb_id]

                # Ensure chain group exists
                if chain_id not in protein_group:
                    chain_group = protein_group.create_group(chain_id)
                else:
                    chain_group = protein_group[chain_id]

                # Check if embedding already exists
                embedding_key = 'esm2_embedding'
                if embedding_key in chain_group:
                    skipped_count += 1
                    continue

                # Store embedding with compression
                embedding = data['embedding']
                embedding_dataset = chain_group.create_dataset(
                    embedding_key,
                    data=embedding,
                    compression='gzip',
                    compression_opts=compression_level
                )

                # Store embedding metadata
                embedding_dataset.attrs['extraction_layer'] = data.get('layer', 'unknown')
                embedding_dataset.attrs['embedding_dim'] = data.get('embedding_dim', embedding.shape[-1])
                embedding_dataset.attrs['extraction_date'] = time.strftime('%Y-%m-%d %H:%M:%S').encode('utf-8')

                # Store sequence if not present
                if 'sequence' not in chain_group:
                    chain_group.create_dataset('sequence', data=data['sequence'])

                # Store contacts if available
                if 'contacts' in data:
                    contacts_dataset = chain_group.create_dataset(
                        'esm2_contacts',
                        data=data['contacts'],
                        compression='gzip',
                        compression_opts=compression_level
                    )
                    contacts_dataset.attrs['extraction_date'] = time.strftime('%Y-%m-%d %H:%M:%S').encode('utf-8')

                # Also store in embeddings group for easy access
                embedding_ref = embeddings_group.create_dataset(identifier, data=embedding)
                embedding_ref.attrs['pdb_id'] = pdb_id.encode('utf-8')
                embedding_ref.attrs['chain_id'] = chain_id.encode('utf-8')
                embedding_ref.attrs['sequence_length'] = embedding.shape[0]
                embedding_ref.attrs['extraction_date'] = time.strftime('%Y-%m-%d %H:%M:%S').encode('utf-8')

                stored_count += 1

            if verbose:
                print(f"✅ Stored {stored_count} new embeddings")
                if skipped_count > 0:
                    print(f"⏭️  Skipped {skipped_count} existing embeddings")

    except Exception as e:
        print(f"❌ Error storing embeddings: {e}")
        raise

def process_hdf5_embeddings(input_hdf5: Path, output_hdf5: Path, model, batch_converter,
                            model_info: Dict, batch_size: int = 8, sample_size: Optional[int] = None,
                            resume: bool = True, compression_level: int = 4, verbose: bool = True,
                            global_progress_callback: Optional[callable] = None):
    """
    Process existing HDF5 dataset and add ESM-2 embeddings.

    Args:
        input_hdf5: Input HDF5 file with contact data
        output_hdf5: Output HDF5 file (can be same as input)
        model: Loaded ESM-2 model
        batch_converter: ESM batch converter
        model_info: Model information dictionary
        batch_size: Batch size for embedding computation
        sample_size: Number of chains to process (None for all)
        resume: Whether to resume if embeddings already exist
        compression_level: Compression level for HDF5 storage
        verbose: Whether to show detailed progress information
        global_progress_callback: Optional callback for global progress updates
    """
    if verbose:
        print(f"🔄 Processing HDF5 dataset: {input_hdf5}")

    if not input_hdf5.exists():
        print(f"❌ Input HDF5 file not found: {input_hdf5}")
        return

    # Use same file if output not specified
    if verbose:
        if output_hdf5 == input_hdf5:
            print(f"📝 Adding embeddings to existing file")
        else:
            print(f"📁 Creating new embeddings file: {output_hdf5}")

    # Load dataset
    dataset = ContactDataset(str(input_hdf5))
    total_chains = len(dataset)

    if verbose:
        print(f"📊 Dataset contains {total_chains} protein chains")

    if sample_size:
        if verbose:
            print(f"📋 Processing sample of {sample_size} chains")
        indices = list(range(min(sample_size, total_chains)))
    else:
        if verbose:
            print(f"📋 Processing all chains")
        indices = list(range(total_chains))

    # Check for existing embeddings
    processed_chains = set()
    if resume and output_hdf5.exists():
        if verbose:
            print(f"🔍 Checking existing embeddings...")
        try:
            def check_embeddings_operation():
                return h5py.File(output_hdf5, 'r')

            with retry_hdf5_operation(check_embeddings_operation, verbose=False) as f:
                if 'esm2_embeddings' in f:
                    embeddings_group = f['esm2_embeddings']
                    for identifier in embeddings_group.keys():
                        processed_chains.add(identifier)
            if verbose:
                print(f"✅ Found embeddings for {len(processed_chains)} chains")
        except Exception as e:
            print(f"⚠️  Error checking existing file: {e}")

    # Process chains
    batch_sequences = {}
    batch_identifiers = []
    processed_count = 0
    failed_count = 0

    # Disable individual progress bars when global tracking is active
    disable_individual = not verbose or global_progress_callback is not None
    progress_desc = "Processing chains" if verbose else None
    for idx in (tqdm(indices, desc=progress_desc, disable=disable_individual) if not disable_individual else indices):
        try:
            # Get chain data
            sequence, contact_map = dataset[idx]
            chain_info = dataset.get_chain_info(idx)

            pdb_id = chain_info['pdb_id']
            chain_id = chain_info['chain_id']
            identifier = f"{pdb_id}_{chain_id}"

            # Skip if already processed
            if resume and identifier in processed_chains:
                # Update global progress for skipped chains
                if global_progress_callback:
                    global_progress_callback(1)
                continue

            # Add to batch
            batch_sequences[chain_id] = sequence
            batch_identifiers.append((pdb_id, chain_id, identifier))

            # Process batch when it reaches desired size
            if len(batch_sequences) >= batch_size or idx == indices[-1]:
                if batch_sequences:
                    # Compute embeddings
                    embeddings = compute_esm2_embeddings(
                        model, batch_converter, batch_sequences,
                        prefix="batch", batch_size=batch_size, verbose=verbose
                    )

                    # Remap identifiers to correct PDB/chain IDs
                    remapped_embeddings = {}
                    for i, (pdb_id, chain_id, identifier) in enumerate(batch_identifiers):
                        batch_identifier = f"batch_{chain_id}"
                        if batch_identifier in embeddings:
                            remapped_embeddings[identifier] = embeddings[batch_identifier]
                            remapped_embeddings[identifier]['sequence'] = batch_sequences[chain_id]

                    # Update global progress for all chains in this batch (both newly processed and stored)
                    if global_progress_callback:
                        global_progress_callback(len(batch_identifiers))

                    if remapped_embeddings:
                        # Store embeddings
                        store_embeddings_hdf5(
                            remapped_embeddings, output_hdf5, model_info,
                            compression_level=compression_level, verbose=verbose
                        )
                        processed_count += len(remapped_embeddings)

                    # Clear batch
                    batch_sequences = {}
                    batch_identifiers = []
                    remapped_embeddings = {}

                    # Memory management
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error processing chain {idx}: {e}")
            failed_count += 1
            continue

    # Summary
    if verbose:
        print(f"\n📊 Processing Summary:")
        print(f"  Chains successfully processed: {processed_count}")
        print(f"  Failed chains: {failed_count}")

    # Get final file info
    if output_hdf5.exists():
        file_size_mb = output_hdf5.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"📁 Output file size: {file_size_mb:.2f} MB")

def get_hdf5_embeddings_info(hdf5_path: Path):
    """
    Get information about embeddings stored in HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Dictionary with embeddings information
    """
    if not hdf5_path.exists():
        return {'error': 'File not found'}

    try:
        def get_info_operation():
            return h5py.File(hdf5_path, 'r')

        with retry_hdf5_operation(get_info_operation, verbose=False) as f:
            info = {}

            # Model information
            if 'esm2_embeddings' in f:
                embeddings_group = f['esm2_embeddings']

                # Decode string attributes
                for attr_name, attr_value in embeddings_group.attrs.items():
                    if isinstance(attr_value, bytes):
                        info[attr_name] = attr_value.decode('utf-8')
                    else:
                        info[attr_name] = attr_value

                # Statistics
                info['total_embeddings'] = len(embeddings_group)

                # Sample identifiers
                identifiers = list(embeddings_group.keys())
                info['sample_identifiers'] = identifiers[:5]

                # Sequence length statistics
                lengths = []
                for identifier in identifiers:
                    if identifier in embeddings_group:
                        length = embeddings_group[identifier].attrs.get('sequence_length', 0)
                        lengths.append(length)

                if lengths:
                    info['length_stats'] = {
                        'mean': float(np.mean(lengths)),
                        'std': float(np.std(lengths)),
                        'min': int(min(lengths)),
                        'max': int(max(lengths))
                    }

            # Dataset structure information
            if 'proteins' in f:
                proteins_group = f['proteins']
                total_proteins = len(proteins_group)
                total_chains_with_embeddings = 0

                for pdb_id in proteins_group:
                    protein_group = proteins_group[pdb_id]
                    for chain_id in protein_group:
                        chain_group = protein_group[chain_id]
                        if 'esm2_embedding' in chain_group:
                            total_chains_with_embeddings += 1

                info['dataset_stats'] = {
                    'total_proteins': total_proteins,
                    'total_chains_with_embeddings': total_chains_with_embeddings
                }

            return info

    except Exception as e:
        return {'error': str(e)}