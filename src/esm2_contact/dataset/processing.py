"""
Processing functions for protein contact prediction datasets.

This module provides functions for processing PDB files and generating
contact maps for protein structure prediction.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser
import h5py
import gc
from pathlib import Path
from typing import Dict, Optional


def load_amino_acid_mapping(mapping_file: str = "amino_acid_three_to_one.json") -> Dict[str, str]:
    """
    Load amino acid 3-letter to 1-letter mapping.

    Args:
        mapping_file: Path to the JSON mapping file

    Returns:
        Dictionary mapping 3-letter codes to 1-letter codes
    """
    with open(mapping_file, 'r') as f:
        return json.load(f)


def extract_chains_from_pdb(pdb_path: Path, aa_three_to_one: Dict[str, str]) -> Dict[str, Dict]:
    """
    Extract individual chains from a PDB file.

    Args:
        pdb_path: Path to PDB file
        aa_three_to_one: Amino acid mapping dictionary

    Returns:
        Dictionary with chain_id as key and chain data as value
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_path))

    chains_data = {}

    # Collect all chains first to get accurate count
    all_chains = []
    for model in structure:
        for chain in model:
            all_chains.append(chain)

    # Progress bar for chain processing (only show for multi-chain files)
    chain_iterator = tqdm(all_chains,
                         desc=f"Processing chains in {pdb_path.name}",
                         leave=False,
                         disable=len(all_chains) <= 1)

    for chain in chain_iterator:
        chain_id = chain.id
        residues = []
        sequence = ""
        ca_coords = []

        for residue in chain:
            # Skip hetero residues and water
            if residue.id[0] != ' ':
                continue

            # Get residue name
            res_name = residue.get_resname()
            if res_name not in aa_three_to_one:
                continue

            # Get CA atom coordinates
            if 'CA' in residue:
                ca_coord = residue['CA'].get_coord()
                res_seq = residue.id[1]  # residue sequence number
                icode = residue.id[2]   # insertion code

                residues.append({
                    'res_name': res_name,
                    'res_seq': res_seq,
                    'icode': icode,
                    'ca_coord': ca_coord
                })

                sequence += aa_three_to_one[res_name]
                ca_coords.append(ca_coord)

        if len(residues) > 0:  # Only add chains with actual residues
            chains_data[chain_id] = {
                'sequence': sequence,
                'residues': residues,
                'ca_coords': np.array(ca_coords),
                'length': len(residues)
            }

    return chains_data


def compute_contact_map(ca_coords: np.ndarray, threshold: float = 8.0,
                        min_seq_separation: int = 5) -> np.ndarray:
    """
    Compute contact map from Cα coordinates.

    Args:
        ca_coords: Array of Cα coordinates with shape (L, 3)
        threshold: Distance threshold in Angstroms for defining contacts
        min_seq_separation: Minimum sequence separation for valid contacts

    Returns:
        Binary contact map with shape (L, L)
    """
    L = len(ca_coords)
    contact_map = np.zeros((L, L), dtype=np.int8)

    # Use progress bar for larger proteins (>100 residues)
    show_progress = L > 100

    if show_progress:
        outer_iterator = tqdm(range(L), desc="Computing contacts", leave=False)
    else:
        outer_iterator = range(L)

    for i in outer_iterator:
        for j in range(i + min_seq_separation, L):  # Only consider i < j with min separation
            # Compute Euclidean distance between CA atoms
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])

            if dist < threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1  # Symmetric matrix

    return contact_map


def process_pdb_dataset(data_path: Path, max_files: Optional[int] = None,
                       aa_three_to_one: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Process all PDB files in a directory and create a dataset.

    Args:
        data_path: Path to directory containing PDB files
        max_files: Maximum number of files to process (for testing)
        aa_three_to_one: Amino acid mapping dictionary (loaded if None)

    Returns:
        DataFrame with essential columns for contact prediction
    """
    if aa_three_to_one is None:
        aa_three_to_one = load_amino_acid_mapping()

    pdb_files = list(data_path.glob("*.pdb"))
    if max_files:
        pdb_files = pdb_files[:max_files]

    dataset_records = []
    failed_files = []

    # Enhanced progress bar with file information
    progress_bar = tqdm(pdb_files,
                       desc=f"Processing {data_path.name} dataset",
                       unit="files",
                       dynamic_ncols=True)

    for pdb_file in progress_bar:
        # Update progress bar description with current file
        progress_bar.set_postfix_str(f"File: {pdb_file.name}")

        try:
            chains_data = extract_chains_from_pdb(pdb_file, aa_three_to_one)

            for chain_id, data in chains_data.items():
                # Skip very short chains
                if data['length'] < 10:
                    continue

                contact_map = compute_contact_map(data['ca_coords'])

                record = {
                    'pdb_id': pdb_file.stem,
                    'chain_id': chain_id,
                    'sequence': data['sequence'],
                    'length': data['length'],
                    'contact_map': contact_map
                }
                dataset_records.append(record)

        except Exception as e:
            failed_files.append((pdb_file.name, str(e)))
            continue

    df = pd.DataFrame(dataset_records)
    print(f"\nProcessed {len(pdb_files)} PDB files")
    print(f"Successfully extracted {len(df)} chains")
    print(f"Failed to process {len(failed_files)} files")

    if failed_files:
        print("Failed files:")
        for filename, error in failed_files[:5]:  # Show first 5 errors
            print(f"  {filename}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    return df


def save_dataset_to_hdf5(df: pd.DataFrame, output_path: Path, batch_size: int = 1000) -> None:
    """
    Save dataset to HDF5 format for efficient loading.

    Args:
        df: DataFrame containing protein chains
        output_path: Path to save HDF5 file
        batch_size: Number of records to process at a time
    """
    print(f"Saving {len(df)} protein chains to {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Create main group
        main_group = f.create_group('proteins')

        # Process in batches to manage memory
        for i in tqdm(range(0, len(df), batch_size), desc="Saving to HDF5"):
            batch_df = df.iloc[i:i+batch_size]

            for idx, row in batch_df.iterrows():
                pdb_id = row['pdb_id']
                chain_id = row['chain_id']

                # Create group for this protein if it doesn't exist
                if pdb_id not in main_group:
                    protein_group = main_group.create_group(pdb_id)
                else:
                    protein_group = main_group[pdb_id]

                # Create group for this chain
                chain_group = protein_group.create_group(chain_id)

                # Store sequence as string dataset
                chain_group.create_dataset('sequence', data=row['sequence'])

                # Store contact map as compressed dataset
                chain_group.create_dataset('contact_map', data=row['contact_map'],
                                         compression='gzip', compression_opts=9)

                # Store metadata as attributes
                chain_group.attrs['length'] = row['length']
                chain_group.attrs['num_contacts'] = np.sum(row['contact_map'])
                chain_group.attrs['contact_density'] = np.sum(row['contact_map']) / (row['length'] * (row['length'] - 1) / 2)

            # Force garbage collection
            del batch_df
            gc.collect()

    print(f"Dataset saved successfully to {output_path}")


def generate_dataset_statistics(train_path: Path, test_path: Path, processed_path: Path) -> Dict:
    """
    Generate comprehensive statistics for the processed datasets.

    Args:
        train_path: Path to training HDF5 file
        test_path: Path to test HDF5 file
        processed_path: Path to processed data directory

    Returns:
        Dictionary with dataset statistics
    """
    stats = {}

    def analyze_hdf5(file_path: Path, dataset_name: str) -> Optional[Dict]:
        """Analyze a single HDF5 dataset."""
        if not file_path.exists():
            return None

        print(f"Analyzing {dataset_name} dataset...")

        lengths = []
        num_contacts = []
        contact_densities = []
        total_chains = 0
        total_proteins = 0

        with h5py.File(file_path, 'r') as f:
            proteins_group = f['proteins']
            total_proteins = len(proteins_group)

            for pdb_id in tqdm(proteins_group, desc=f"Scanning {dataset_name}"):
                protein_group = proteins_group[pdb_id]
                for chain_id in protein_group:
                    chain_group = protein_group[chain_id]
                    total_chains += 1
                    lengths.append(chain_group.attrs['length'])
                    num_contacts.append(chain_group.attrs['num_contacts'])
                    contact_densities.append(chain_group.attrs['contact_density'])

        return {
            'total_chains': total_chains,
            'total_proteins': total_proteins,
            'avg_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'avg_contacts': float(np.mean(num_contacts)),
            'median_contacts': float(np.median(num_contacts)),
            'avg_contact_density': float(np.mean(contact_densities))
        }

    # Analyze training dataset (only if it exists)
    if train_path.exists():
        stats['train'] = analyze_hdf5(train_path, "training")
    else:
        print(f"Training dataset not found at {train_path}")

    # Analyze test dataset
    if test_path.exists():
        stats['test'] = analyze_hdf5(test_path, "test")
    else:
        print(f"Test dataset not found at {test_path}")
        return None

    # Save statistics
    if stats:
        stats_path = processed_path / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDataset statistics saved to {stats_path}")

        # Print summary
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        for split in stats:
            print(f"\n{split.upper()} SET:")
            for key, value in stats[split].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

    return stats