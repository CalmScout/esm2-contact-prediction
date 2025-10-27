"""
Dataset Loader for Tiny10 CNN Training Data

This module provides a PyTorch Dataset class for loading the synchronized tiny_10
CNN dataset with 68-channel inputs (4 template + 64 ESM2 channels) and binary
contact map targets.

Key Features:
- Memory-efficient loading of CNN dataset
- Support for variable-sized proteins
- Automatic data validation and cleaning
- Fast initialization without heavy data loading
- Binary contact map preprocessing

Dataset Structure:
- Input: 68-channel tensors (4 template + 64 ESM2 channels)
- Target: Binary contact maps (0/1 values)
- Size: 10 proteins total
- Format: HDF5 file with metadata

Usage:
    dataset = Tiny10Dataset("data/tiny_10/cnn_dataset.h5")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class Tiny10Dataset(Dataset):
    """
    Dataset for loading the tiny_10 CNN training data with binary contact maps.

    This dataset loads multi-channel input features (68 channels) and binary contact
    map targets from the synchronized tiny_10 dataset created by the pipeline.

    Args:
        cnn_file (str): Path to the CNN dataset HDF5 file
        validate_data (bool): Whether to validate data integrity (default: True)
        verbose (bool): Whether to print verbose information (default: True)

    Attributes:
        cnn_file (str): Path to the CNN dataset file
        query_ids (List[str]): List of valid query IDs in the dataset
        verbose (bool): Verbose output flag
    """

    def __init__(self, cnn_file: str, validate_data: bool = True, verbose: bool = True):
        self.cnn_file = Path(cnn_file)
        self.verbose = verbose

        if not self.cnn_file.exists():
            raise FileNotFoundError(f"CNN dataset file not found: {self.cnn_file}")

        # Get available query IDs
        with h5py.File(self.cnn_file, 'r') as f:
            if 'cnn_data' not in f:
                raise ValueError(f"Invalid CNN dataset format: missing 'cnn_data' group")
            self.query_ids = list(f['cnn_data'].keys())

        if self.verbose:
            print(f"📊 Found {len(self.query_ids)} total proteins in CNN dataset")

        # Validate and filter valid proteins
        if validate_data:
            self.query_ids = self._validate_proteins(self.query_ids)

        if len(self.query_ids) == 0:
            raise ValueError("No valid proteins found in dataset. Check data integrity.")

        if self.verbose:
            print(f"✅ Using {len(self.query_ids)} valid proteins")
            self._print_dataset_info()

    def _validate_proteins(self, query_ids: List[str]) -> List[str]:
        """Validate proteins and return list of valid IDs."""
        valid_ids = []

        with h5py.File(self.cnn_file, 'r') as f:
            for qid in query_ids:
                try:
                    query_group = f['cnn_data'][qid]

                    # Check required fields exist
                    required_fields = ['multi_channel_input', 'consensus_contact_map']
                    if not all(field in query_group for field in required_fields):
                        if self.verbose:
                            print(f"   ⚠️  Skipping {qid}: missing required fields")
                        continue

                    # Check sequence length attribute
                    seq_len = query_group.attrs.get('sequence_length', 0)
                    if seq_len <= 0:
                        if self.verbose:
                            print(f"   ⚠️  Skipping {qid}: invalid sequence length")
                        continue

                    # Quick data shape validation (no loading)
                    features_shape = query_group['multi_channel_input'].shape
                    contacts_shape = query_group['consensus_contact_map'].shape

                    if len(features_shape) != 3 or features_shape[0] != 68:
                        if self.verbose:
                            print(f"   ⚠️  Skipping {qid}: invalid features shape {features_shape}")
                        continue

                    if len(contacts_shape) != 2:
                        if self.verbose:
                            print(f"   ⚠️  Skipping {qid}: invalid contacts shape {contacts_shape}")
                        continue

                    valid_ids.append(qid)

                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  Skipping {qid}: validation error - {e}")
                    continue

        return valid_ids

    def _print_dataset_info(self):
        """Print dataset information."""
        if not self.verbose or len(self.query_ids) == 0:
            return

        try:
            # Load first sample for info
            sample = self[0]
            print(f"📈 Dataset Information:")
            print(f"   Total proteins: {len(self)}")
            print(f"   Sample protein: {sample['query_id']}")
            print(f"   Sample length: {sample['length']} residues")
            print(f"   Feature channels: {sample['features'].shape[0]}")
            print(f"   Feature shape: {sample['features'].shape}")
            print(f"   Contact density: {torch.mean(sample['contacts']):.4f}")
            print(f"   Unique contact values: {torch.unique(sample['contacts']).tolist()}")

            # Basic statistics
            lengths = []
            densities = []
            for i in range(min(len(self), 5)):  # Sample first 5 proteins
                data = self[i]
                lengths.append(data['length'])
                densities.append(torch.mean(data['contacts']).item())

            print(f"   Sample lengths: {lengths}")
            print(f"   Sample densities: {[f'{d:.4f}' for d in densities]}")

        except Exception as e:
            print(f"   ⚠️  Could not display detailed info: {e}")

    def __len__(self) -> int:
        """Return the number of proteins in the dataset."""
        return len(self.query_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'features': Multi-channel input tensor (channels, H, W)
                - 'contacts': Binary contact map tensor (H, W)
                - 'length': Sequence length (int)
                - 'query_id': Protein identifier (str)
        """
        query_id = self.query_ids[idx]

        try:
            # Load data from CNN dataset
            with h5py.File(self.cnn_file, 'r') as f:
                query_group = f['cnn_data'][query_id]

                # Load CNN features
                features = query_group['multi_channel_input'][:]
                seq_len = query_group.attrs.get('sequence_length', features.shape[1])

                # Load contact map
                contacts = query_group['consensus_contact_map'][:]

                # Validate and clean contact map
                contacts = self._clean_contact_map(contacts, seq_len)

            # Ensure matching sizes
            min_len = min(features.shape[1], contacts.shape[0], seq_len)
            features = features[:, :min_len, :min_len]
            contacts = contacts[:min_len, :min_len]

            return {
                'features': torch.FloatTensor(features),
                'contacts': torch.FloatTensor(contacts),
                'length': int(min_len),
                'query_id': query_id
            }

        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} ({query_id}): {e}")

    def _clean_contact_map(self, contacts: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Clean and validate contact map.

        Args:
            contacts (np.ndarray): Raw contact map
            seq_len (int): Expected sequence length

        Returns:
            np.ndarray: Cleaned binary contact map
        """
        # Ensure proper size
        contacts = contacts[:seq_len, :seq_len]

        # Ensure binary values (0 or 1) - use proper threshold
        contacts = (contacts > 0.5).astype(np.float32)

        # Remove diagonal (self-contacts)
        np.fill_diagonal(contacts, 0)

        # Make symmetric (ensure contact map is symmetric)
        contacts = np.maximum(contacts, contacts.T)

        return contacts

    def get_dataset_statistics(self) -> Dict:
        """Get comprehensive dataset statistics."""
        if len(self) == 0:
            return {}

        lengths = []
        densities = []
        contact_counts = []

        print("📊 Computing dataset statistics...")

        for i in range(len(self)):
            try:
                sample = self[i]
                length = sample['length']
                contacts = sample['contacts']

                lengths.append(length)
                density = torch.mean(contacts).item()
                densities.append(density)
                contact_counts.append(torch.sum(contacts).item())

            except Exception as e:
                print(f"   ⚠️  Error processing sample {i}: {e}")
                continue

        if not lengths:
            return {}

        stats = {
            'total_proteins': len(lengths),
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths)
            },
            'density_stats': {
                'min': min(densities),
                'max': max(densities),
                'mean': np.mean(densities),
                'std': np.std(densities)
            },
            'contact_stats': {
                'total_contacts': sum(contact_counts),
                'avg_contacts_per_protein': np.mean(contact_counts),
                'contact_range': [min(contact_counts), max(contact_counts)]
            }
        }

        print(f"✅ Dataset statistics computed:")
        print(f"   Proteins: {stats['total_proteins']}")
        print(f"   Length range: {stats['length_stats']['min']}-{stats['length_stats']['max']}")
        print(f"   Density range: {stats['density_stats']['min']:.4f}-{stats['density_stats']['max']:.4f}")
        print(f"   Total contacts: {stats['contact_stats']['total_contacts']}")

        return stats


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-sized proteins with proper padding.

    Args:
        batch (List[Dict]): List of samples from the dataset

    Returns:
        Tuple containing:
        - features (torch.Tensor): Padded features (batch_size, channels, max_len, max_len)
        - contacts (torch.Tensor): Padded contact maps (batch_size, max_len, max_len)
        - mask (torch.Tensor): Valid region mask (batch_size, max_len, max_len)
        - lengths (torch.Tensor): Original sequence lengths (batch_size,)
    """
    if len(batch) == 0:
        raise ValueError("Empty batch provided to collate_fn")

    # Get maximum length in batch
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    channels = batch[0]['features'].shape[0]

    # Create padded tensors
    features = torch.zeros(batch_size, channels, max_len, max_len)
    contacts = torch.zeros(batch_size, max_len, max_len)
    mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool)
    lengths = []

    for i, item in enumerate(batch):
        l = item['length']
        features[i, :, :l, :l] = item['features']
        contacts[i, :l, :l] = item['contacts']
        mask[i, :l, :l] = True
        lengths.append(l)

    return features, contacts, mask, torch.tensor(lengths)


def create_data_splits(dataset: Tiny10Dataset, train_ratio: float = 0.8,
                      val_ratio: float = 0.1, test_ratio: float = 0.1,
                      random_seed: int = 42) -> Tuple[Tiny10Dataset, Tiny10Dataset, Tiny10Dataset]:
    """
    Create train/validation/test splits from the dataset.

    Args:
        dataset (Tiny10Dataset): Input dataset
        train_ratio (float): Training set ratio (default: 0.8)
        val_ratio (float): Validation set ratio (default: 0.1)
        test_ratio (float): Test set ratio (default: 0.1)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple containing train_dataset, val_dataset, test_dataset
    """
    total_size = len(dataset)

    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure all data is used

    print(f"🎯 Creating data splits:")
    print(f"   Total proteins:     {total_size}")
    print(f"   Training proteins:  {train_size} ({train_ratio*100:.0f}%)")
    print(f"   Validation proteins: {val_size} ({val_ratio*100:.0f}%)")
    print(f"   Test proteins:       {test_size} ({test_ratio/total_size*100:.1f}%)")

    # Create reproducible generator
    generator = torch.Generator().manual_seed(random_seed)

    # Create splits
    from torch.utils.data import random_split

    # First split: train + val vs test
    train_val_dataset, test_dataset = random_split(
        dataset,
        [train_size + val_size, test_size],
        generator=generator
    )

    # Second split: train vs val from train_val
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"✅ Data splits created successfully!")
    return train_dataset, val_dataset, test_dataset


# Test function for dataset validation
def test_dataset():
    """Test dataset functionality."""
    print("🧪 Testing Tiny10Dataset...")

    # This will fail if the dataset file doesn't exist
    dataset_path = "data/tiny_10/cnn_dataset.h5"

    if not Path(dataset_path).exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        print("   Run the pipeline first to generate the dataset.")
        return None

    try:
        # Create dataset
        dataset = Tiny10Dataset(dataset_path, verbose=True)
        print(f"✅ Dataset loaded successfully")

        # Test data loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded:")
            print(f"   Query ID: {sample['query_id']}")
            print(f"   Features shape: {sample['features'].shape}")
            print(f"   Contacts shape: {sample['contacts'].shape}")
            print(f"   Length: {sample['length']}")
            print(f"   Contact density: {torch.mean(sample['contacts']):.4f}")

            # Test collate function
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
            batch = next(iter(dataloader))
            features, contacts, mask, lengths = batch
            print(f"✅ Collate function test:")
            print(f"   Batch features shape: {features.shape}")
            print(f"   Batch contacts shape: {contacts.shape}")
            print(f"   Batch mask shape: {mask.shape}")
            print(f"   Batch lengths: {lengths.tolist()}")

        # Test data splits
        if len(dataset) >= 3:
            train_ds, val_ds, test_ds = create_data_splits(dataset)
            print(f"✅ Data splits created: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

        # Test statistics
        stats = dataset.get_dataset_statistics()

        print("🎉 Dataset test completed successfully!")
        return dataset

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return None


if __name__ == "__main__":
    # Run dataset test
    test_dataset()