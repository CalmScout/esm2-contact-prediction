"""
Homology search and template processing for protein contact prediction.

This module provides tools for finding homologous protein structures,
aligning sequences, and processing template information for
homology-assisted contact prediction.
"""

from .search import TemplateSearcher, DatabaseConfig, DualSearchResult, TemplateSearchResult
from .alignment import SequenceAligner
from .template_db import TemplateDatabase
from .robust_processor import RobustTemplateProcessor

__all__ = [
    'TemplateSearcher',
    'DatabaseConfig',
    'DualSearchResult',
    'TemplateSearchResult',
    'SequenceAligner',
    'TemplateDatabase',
    'RobustTemplateProcessor'
]