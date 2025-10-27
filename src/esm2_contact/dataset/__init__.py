"""
Dataset module for protein contact prediction.

This module provides utilities for loading and processing protein contact
prediction datasets from HDF5 files.
"""

from .utils import ContactDataset, collate_contact_maps, load_dataset_info
from .processing import (
    extract_chains_from_pdb,
    compute_contact_map,
    process_pdb_dataset,
    save_dataset_to_hdf5,
    generate_dataset_statistics
)

__all__ = [
    "ContactDataset",
    "collate_contact_maps",
    "load_dataset_info",
    "extract_chains_from_pdb",
    "compute_contact_map",
    "process_pdb_dataset",
    "save_dataset_to_hdf5",
    "generate_dataset_statistics"
]