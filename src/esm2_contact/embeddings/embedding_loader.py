"""
ESM2 Embedding Loader Module

This module provides utilities for loading and managing ESM2 embeddings
pre-computed and stored in HDF5 format.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import h5py
import numpy as np

logger = logging.getLogger(__name__)


class ESM2EmbeddingLoader:
    """
    Utility class for loading ESM2 embeddings from pre-computed HDF5 files.

    This class handles loading of ESM2-650M embeddings that were pre-computed
    in notebook 03_esm2_embeddings.ipynb and stored in train_with_embeddings.h5
    and test_with_embeddings.h5 files.
    """

    def __init__(self, data_dir: Path, cache_embeddings: bool = True):
        """
        Initialize the ESM2 embedding loader.

        Args:
            data_dir: Directory containing the embedding files
            cache_embeddings: Whether to cache embeddings in memory for efficiency
        """
        self.data_dir = Path(data_dir)
        self.cache_embeddings = cache_embeddings
        self._embedding_cache = {} if cache_embeddings else None
        self._file_handles = {}  # Keep file handles open for efficiency

        # Initialize file handles
        self._train_file = None
        self._test_file = None
        self._initialize_files()

        logger.info(f"Initialized ESM2EmbeddingLoader with data_dir: {self.data_dir}")

    def _initialize_files(self):
        """Initialize file handles for embedding files."""
        train_path = self.data_dir / "train_with_embeddings.h5"
        test_path = self.data_dir / "test_with_embeddings.h5"

        if train_path.exists():
            self._train_file = h5py.File(train_path, 'r')
            logger.info(f"Opened train embeddings file: {train_path}")

        if test_path.exists():
            self._test_file = h5py.File(test_path, 'r')
            logger.info(f"Opened test embeddings file: {test_path}")

        if not self._train_file and not self._test_file:
            raise FileNotFoundError(
                f"No embedding files found in {self.data_dir}. "
                f"Expected train_with_embeddings.h5 and/or test_with_embeddings.h5"
            )

    def _get_protein_file(self, protein_id: str) -> Optional[h5py.File]:
        """
        Determine which file contains the given protein ID.

        Args:
            protein_id: Protein identifier (e.g., "106M_A")

        Returns:
            HDF5 file handle containing the protein, or None if not found
        """
        # Check train file first
        if self._train_file and 'esm2_embeddings' in self._train_file:
            if protein_id in self._train_file['esm2_embeddings']:
                return self._train_file

        # Check test file
        if self._test_file and 'esm2_embeddings' in self._test_file:
            if protein_id in self._test_file['esm2_embeddings']:
                return self._test_file

        return None

    def get_embedding_info(self, protein_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about available embeddings for a protein.

        Args:
            protein_id: Protein identifier

        Returns:
            Dictionary with embedding info or None if not found
        """
        file_handle = self._get_protein_file(protein_id)
        if not file_handle:
            return None

        try:
            embeddings_group = file_handle['esm2_embeddings']
            if protein_id in embeddings_group:
                embeddings = embeddings_group[protein_id]
                return {
                    'protein_id': protein_id,
                    'shape': embeddings.shape,
                    'dtype': embeddings.dtype,
                    'file_source': 'train' if file_handle == self._train_file else 'test'
                }
        except Exception as e:
            logger.error(f"Error getting embedding info for {protein_id}: {e}")

        return None

    def load_embeddings(self, protein_id: str, sequence: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load ESM2 embeddings for a specific protein.

        Args:
            protein_id: Protein identifier (e.g., "106M_A")
            sequence: Optional protein sequence for validation

        Returns:
            ESM2 embeddings array of shape (L, 1280) or None if not found
        """
        # Check cache first
        if self._embedding_cache and protein_id in self._embedding_cache:
            cached_embedding = self._embedding_cache[protein_id]
            if sequence is None or len(cached_embedding) == len(sequence):
                logger.debug(f"Using cached embeddings for {protein_id}")
                return cached_embedding
            else:
                logger.warning(f"Sequence length mismatch for cached {protein_id}, reloading")

        # Find the file containing this protein
        file_handle = self._get_protein_file(protein_id)
        if not file_handle:
            logger.warning(f"Protein {protein_id} not found in embedding files")
            return None

        try:
            embeddings_group = file_handle['esm2_embeddings']
            if protein_id not in embeddings_group:
                logger.warning(f"Protein {protein_id} not found in embeddings")
                return None

            embeddings = embeddings_group[protein_id][:]

            # Validate against sequence if provided
            if sequence is not None:
                if len(embeddings) != len(sequence):
                    logger.warning(
                        f"Embedding length mismatch for {protein_id}: "
                        f"embeddings={len(embeddings)} vs sequence={len(sequence)}"
                    )
                    # We can still proceed, but log the discrepancy

            # Cache the embeddings if enabled
            if self._embedding_cache is not None:
                self._embedding_cache[protein_id] = embeddings

            logger.debug(f"Loaded embeddings for {protein_id}: shape={embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Error loading embeddings for {protein_id}: {e}")
            return None

    def load_embeddings_for_sequence(self, protein_id: str, target_sequence: str) -> Optional[np.ndarray]:
        """
        Load ESM2 embeddings and align them to a target sequence.

        This method handles cases where the pre-computed embeddings might have
        different sequence lengths than the target sequence due to processing
        differences.

        Args:
            protein_id: Protein identifier
            target_sequence: Target protein sequence to align to

        Returns:
            Aligned embeddings array or None if loading failed
        """
        embeddings = self.load_embeddings(protein_id, target_sequence)
        if embeddings is None:
            return None

        # If lengths match, return as-is
        if len(embeddings) == len(target_sequence):
            return embeddings

        # If lengths don't match, try simple truncation or padding
        if len(embeddings) > len(target_sequence):
            # Truncate to target length
            aligned_embeddings = embeddings[:len(target_sequence)]
            logger.warning(
                f"Truncated embeddings for {protein_id}: "
                f"{len(embeddings)} -> {len(target_sequence)}"
            )
        else:
            # Pad with zeros
            aligned_embeddings = np.zeros((len(target_sequence), embeddings.shape[1]), dtype=embeddings.dtype)
            aligned_embeddings[:len(embeddings)] = embeddings
            logger.warning(
                f"Padded embeddings for {protein_id}: "
                f"{len(embeddings)} -> {len(target_sequence)}"
            )

        return aligned_embeddings

    def list_available_proteins(self) -> Dict[str, list]:
        """
        List all available proteins in the embedding files.

        Returns:
            Dictionary with 'train' and 'test' keys containing protein lists
        """
        proteins = {'train': [], 'test': []}

        if self._train_file and 'esm2_embeddings' in self._train_file:
            proteins['train'] = list(self._train_file['esm2_embeddings'].keys())

        if self._test_file and 'esm2_embeddings' in self._test_file:
            proteins['test'] = list(self._test_file['esm2_embeddings'].keys())

        return proteins

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        stats = {
            'total_proteins': 0,
            'train_proteins': 0,
            'test_proteins': 0,
            'cached_embeddings': len(self._embedding_cache) if self._embedding_cache else 0,
            'files_loaded': []
        }

        proteins = self.list_available_proteins()
        stats['train_proteins'] = len(proteins['train'])
        stats['test_proteins'] = len(proteins['test'])
        stats['total_proteins'] = stats['train_proteins'] + stats['test_proteins']

        if self._train_file:
            stats['files_loaded'].append('train_with_embeddings.h5')
        if self._test_file:
            stats['files_loaded'].append('test_with_embeddings.h5')

        return stats

    def clear_cache(self):
        """Clear the embedding cache."""
        if self._embedding_cache:
            self._embedding_cache.clear()
            logger.info("Cleared embedding cache")

    def close(self):
        """Close file handles and clean up resources."""
        if self._train_file:
            self._train_file.close()
            self._train_file = None

        if self._test_file:
            self._test_file.close()
            self._test_file = None

        self.clear_cache()
        logger.info("Closed ESM2 embedding loader")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()