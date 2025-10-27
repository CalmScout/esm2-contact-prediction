"""
Dataset utilities for protein contact prediction.

This module provides utilities for loading and working with the processed
protein contact prediction datasets in HDF5 format.
"""

import h5py
import torch
from typing import Dict, Tuple, List
import numpy as np


class ContactDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading protein contact prediction data from HDF5 files.

    This dataset loads protein sequences and their corresponding contact maps
    that have been pre-computed from PDB structures.
    """

    def __init__(self, hdf5_path: str):
        """
        Initialize the dataset.

        Args:
            hdf5_path: Path to HDF5 file containing the dataset
        """
        self.hdf5_path = hdf5_path
        self.h5_file = h5py.File(hdf5_path, 'r')

        # Build index of all protein-chain pairs for fast random access
        self.index = []
        proteins_group = self.h5_file['proteins']

        for pdb_id in proteins_group:
            protein_group = proteins_group[pdb_id]
            for chain_id in protein_group:
                self.index.append((pdb_id, chain_id))

    def __len__(self):
        """Return the total number of protein chains in the dataset."""
        return len(self.index)

    def __getitem__(self, idx):
        """
        Get a single protein chain data.

        Args:
            idx: Index of the protein chain

        Returns:
            tuple: (sequence_string, contact_map_matrix)
        """
        pdb_id, chain_id = self.index[idx]
        chain_group = self.h5_file['proteins'][pdb_id][chain_id]

        # Load sequence (stored as bytes, decode to string)
        sequence = chain_group['sequence'][()].decode('utf-8')

        # Load contact map
        contact_map = chain_group['contact_map'][()]

        return sequence, contact_map.astype(np.float32)

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5_file') and self.h5_file:
            self.h5_file.close()

    def get_chain_info(self, idx) -> Dict:
        """
        Get metadata for a specific chain.

        Args:
            idx: Index of the protein chain

        Returns:
            Dictionary with metadata about the chain
        """
        pdb_id, chain_id = self.index[idx]
        chain_group = self.h5_file['proteins'][pdb_id][chain_id]

        return {
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'length': chain_group.attrs['length'],
            'num_contacts': chain_group.attrs['num_contacts'],
            'contact_density': chain_group.attrs['contact_density']
        }

    def get_chain_by_id(self, pdb_id: str, chain_id: str) -> Tuple[str, np.ndarray]:
        """
        Get a specific chain by its PDB ID and chain ID.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            tuple: (sequence_string, contact_map_matrix)

        Raises:
            KeyError: If the specified chain is not found
        """
        if pdb_id not in self.h5_file['proteins']:
            raise KeyError(f"PDB ID {pdb_id} not found in dataset")

        protein_group = self.h5_file['proteins'][pdb_id]
        if chain_id not in protein_group:
            raise KeyError(f"Chain {chain_id} not found in PDB {pdb_id}")

        chain_group = protein_group[chain_id]
        sequence = chain_group['sequence'][()].decode('utf-8')
        contact_map = chain_group['contact_map'][()]

        return sequence, contact_map.astype(np.float32)

    def get_statistics(self) -> Dict:
        """
        Get basic statistics about the dataset.

        Returns:
            Dictionary with dataset statistics
        """
        if len(self.index) == 0:
            return {}

        lengths = []
        num_contacts = []
        contact_densities = []

        # Sample a subset for statistics (avoid loading entire dataset)
        sample_size = min(1000, len(self.index))
        sample_indices = np.random.choice(len(self.index), sample_size, replace=False)

        for idx in sample_indices:
            info = self.get_chain_info(idx)
            lengths.append(info['length'])
            num_contacts.append(info['num_contacts'])
            contact_densities.append(info['contact_density'])

        return {
            'total_chains': len(self.index),
            'total_proteins': len(set(pdb_id for pdb_id, _ in self.index)),
            'avg_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'avg_contacts': float(np.mean(num_contacts)),
            'avg_contact_density': float(np.mean(contact_densities))
        }


def collate_contact_maps(batch):
    """
    Collate function for DataLoader to handle variable-sized contact maps.

    Args:
        batch: List of (sequence, contact_map) tuples

    Returns:
        tuple: (sequences_list, contact_maps_tensor, lengths_tensor, mask_tensor)
    """
    sequences, contact_maps = zip(*batch)

    # Get max sequence length in the batch
    max_length = max(len(cm) for cm in contact_maps)
    batch_size = len(contact_maps)

    # Create padded contact maps tensor
    padded_contact_maps = torch.zeros(batch_size, max_length, max_length)
    mask = torch.zeros(batch_size, max_length, max_length, dtype=torch.bool)
    lengths = torch.tensor([len(cm) for cm in contact_maps])

    for i, contact_map in enumerate(contact_maps):
        L = contact_map.shape[0]
        padded_contact_maps[i, :L, :L] = torch.from_numpy(contact_map)
        mask[i, :L, :L] = True

    return sequences, padded_contact_maps, lengths, mask


def load_dataset_info(hdf5_path: str) -> Dict:
    """
    Load basic information about an HDF5 dataset without loading the entire file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Dictionary with dataset information
    """
    info = {
        'total_chains': 0,
        'total_proteins': 0,
        'pdb_ids': []
    }

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'proteins' not in f:
                return {'error': 'Invalid HDF5 file structure'}

            proteins_group = f['proteins']
            info['total_proteins'] = len(proteins_group)

            for pdb_id in proteins_group:
                info['pdb_ids'].append(pdb_id)
                protein_group = proteins_group[pdb_id]
                info['total_chains'] += len(protein_group)

    except Exception as e:
        return {'error': f'Failed to load dataset: {str(e)}'}

    return info