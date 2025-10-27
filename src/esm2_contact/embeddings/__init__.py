"""
ESM2 Embeddings Module

This module provides utilities for loading and managing ESM2 embeddings
pre-computed and stored in HDF5 format.
"""

from .embedding_loader import ESM2EmbeddingLoader

__all__ = ['ESM2EmbeddingLoader']