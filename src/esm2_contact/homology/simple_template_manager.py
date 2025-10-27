"""
Simplified template manager for direct PDB file operations.

This module provides a lightweight alternative to the SQLite database-based
template management, using direct file system operations for template
coordinate extraction and processing.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import numpy as np
from Bio.PDB import PDBParser, PDBList
from Bio import SeqIO
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleTemplateInfo:
    """Lightweight template information class."""

    def __init__(self, pdb_id: str, chain_id: str, sequence: str,
                 length: int, resolution: Optional[float] = None,
                 method: Optional[str] = None, title: Optional[str] = None,
                 deposition_date: Optional[str] = None, file_path: Optional[Path] = None):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.sequence = sequence
        self.length = length
        self.resolution = resolution
        self.method = method
        self.title = title
        self.deposition_date = deposition_date
        self.file_path = file_path
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pdb_id': self.pdb_id,
            'chain_id': self.chain_id,
            'sequence': self.sequence,
            'length': self.length,
            'resolution': self.resolution,
            'method': self.method,
            'title': self.title,
            'deposition_date': self.deposition_date,
            'file_path': str(self.file_path) if self.file_path else None,
            'last_accessed': self.last_accessed.isoformat()
        }


class SimpleTemplateManager:
    """
    Simplified template manager using direct file operations.

    This class provides template functionality without database overhead,
    using direct PDB file parsing and simple in-memory caching.
    """

    def __init__(self, cache_dir: Optional[Path] = None,
                 max_cache_size_gb: float = 10.0):
        """
        Initialize simple template manager.

        Args:
            cache_dir: Directory to cache template files
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = cache_dir or Path.home() / ".esm2_contact" / "template_cache"
        self.max_cache_size_gb = max_cache_size_gb
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # PDB tools
        self.pdb_downloader = PDBList()
        self.parser = PDBParser(QUIET=True)

        # In-memory cache for coordinates during session
        self._coordinate_cache = {}

        logger.info(f"Initialized SimpleTemplateManager with cache: {self.cache_dir}")

    def download_template(self, pdb_id: str) -> Optional[Path]:
        """
        Download a template PDB file if not already cached.

        Args:
            pdb_id: PDB identifier (case-insensitive)

        Returns:
            Path to downloaded PDB file or None if failed
        """
        pdb_id = pdb_id.upper()
        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        if pdb_file.exists():
            return pdb_file

        # COMPLETE suppression of ALL output during download
        import logging
        import os
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO

        # Save current logger configuration
        original_handlers = []
        for handler in logging.root.handlers[:]:
            original_handlers.append((handler, handler.level))
            logging.root.removeHandler(handler)

        # Set very high logging level to suppress everything
        logging.getLogger().setLevel(logging.CRITICAL)

        # Suppress ALL BioPython related loggers comprehensively
        loggers_to_suppress = [
            'Bio', 'Bio.PDB', 'Bio.PDB.PDBList', 'Bio.PDB.PDBIO', 'Bio.PDB.mmtf',
            'Bio.PDB.Parser', 'Bio.PDB.Dice', 'Bio.PDB.StructureBuilder',
            'Bio.PDB.Atom', 'Bio.PDB.Superimposer', 'Bio.PDB.vectors',
            'Bio.PDB.Polypeptide', 'Bio.PDB.PDBExceptions', 'Bio.PDB.Selection',
            'Bio.PDB.ResidueDepth', 'Bio.PDB.HSExposure', 'Bio.PDB.DSSP',
            'Bio.PDB.KDTree', 'Bio.PDB.NeighborSearch', 'Bio.PDB.PDBIO'
        ]
        original_levels = {}
        for logger_name in loggers_to_suppress:
            logger_obj = logging.getLogger(logger_name)
            original_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(logging.CRITICAL)

        # Capture stdout/stderr to suppress any print statements
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            # Complete suppression of ALL output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Download to temporary location first
                self.pdb_downloader.retrieve_pdb_file(pdb_id, pdir=str(self.cache_dir),
                                                     file_format="pdb")

                # PDBList downloads with .ent extension, rename to .pdb
                downloaded_file = self.cache_dir / f"pdb{pdb_id.lower()}.ent"
                if downloaded_file.exists():
                    downloaded_file.rename(pdb_file)
                    return pdb_file
                else:
                    # Try alternative download method (also suppressed)
                    return self._download_template_fallback(pdb_id)

        except Exception as e:
            # Complete silence - only log to file if absolutely critical
            # No console output whatsoever
            return None
        finally:
            # Restore original logging configuration
            for handler, level in original_handlers:
                logging.root.addHandler(handler)
                handler.setLevel(level)

            # Restore BioPython logger levels
            for logger_name, original_level in original_levels.items():
                logger_obj = logging.getLogger(logger_name)
                logger_obj.setLevel(original_level)

            # Clean up captured output (discard it)
            stdout_capture.close()
            stderr_capture.close()

    def _download_template_fallback(self, pdb_id: str) -> Optional[Path]:
        """
        Fallback download method using HTTP requests.

        Args:
            pdb_id: PDB identifier

        Returns:
            Path to downloaded PDB file or None if failed
        """
        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        try:
            # Completely silent HTTP download
            import logging
            from contextlib import redirect_stdout, redirect_stderr
            from io import StringIO

            # Also suppress requests logging
            requests_logger = logging.getLogger('requests.packages.urllib3')
            original_level = requests_logger.level
            requests_logger.setLevel(logging.CRITICAL)

            stdout_capture = StringIO()
            stderr_capture = StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                with open(pdb_file, 'w') as f:
                    f.write(response.text)

            # Clean up
            stdout_capture.close()
            stderr_capture.close()
            requests_logger.setLevel(original_level)

            return pdb_file

        except Exception:
            # Complete silence - no error messages
            return None

    def get_template_info(self, pdb_id: str, chain_id: str) -> Optional[SimpleTemplateInfo]:
        """
        Get template information by parsing PDB file directly.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            SimpleTemplateInfo object or None if not found
        """
        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        if not pdb_file.exists():
            # Try to download it
            if not self.download_template(pdb_id):
                return None
            pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        try:
            # Parse PDB header information
            structure = self.parser.get_structure('template', str(pdb_file))

            # Get first model
            model = next(structure.get_models())

            # Find the requested chain
            if chain_id not in model:
                logger.warning(f"Chain {chain_id} not found in {pdb_id}")
                return None

            chain = model[chain_id]

            # Extract sequence and basic info
            sequence = ""
            for residue in chain:
                if residue.id[0] == ' ':  # Standard amino acid
                    try:
                        aa = residue.get_resname()
                        # Convert three-letter code to one-letter
                        from Bio.Data import IUPACData
                        # Convert to proper case: first letter uppercase, rest lowercase
                        aa_proper = aa[0].upper() + aa[1:].lower() if len(aa) > 1 else aa.upper()
                        aa_one = IUPACData.protein_letters_3to1.get(aa_proper, 'X')
                        sequence += aa_one
                    except:
                        sequence += 'X'

            # Extract header information
            resolution = None
            method = None
            title = None
            deposition_date = None

            # Try to get metadata from PDB header
            for record in SeqIO.parse(str(pdb_file), "pdb-seqres"):
                if record.annotations:
                    resolution = record.annotations.get('resolution')
                    method = record.annotations.get('structure_method')
                    title = record.annotations.get('name')
                    deposition_date = record.annotations.get('date')
                break  # Just get first record

            template_info = SimpleTemplateInfo(
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=sequence,
                length=len(sequence),
                resolution=resolution,
                method=method,
                title=title,
                deposition_date=deposition_date,
                file_path=pdb_file
            )

            return template_info

        except Exception as e:
            logger.error(f"Failed to parse template {pdb_id}_{chain_id}: {e}")
            return None

    def get_template_coordinates(self, pdb_id: str, chain_id: str) -> Optional[np.ndarray]:
        """
        Extract Cα coordinates from a template structure.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            Array of Cα coordinates (L, 3) or None if failed
        """
        cache_key = f"{pdb_id}_{chain_id}"

        # Check in-memory cache first
        if cache_key in self._coordinate_cache:
            return self._coordinate_cache[cache_key]

        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        if not pdb_file.exists():
            # Try to download it
            if not self.download_template(pdb_id):
                return None
            pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        try:
            structure = self.parser.get_structure('template', str(pdb_file))
            ca_coords = []

            # Only use first model
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            if residue.id[0] == ' ':  # Standard amino acid
                                if 'CA' in residue:
                                    ca_coords.append(residue['CA'].get_coord())
                        break  # Found the chain, break chain loop
                break  # Only use first model

            if not ca_coords:
                logger.warning(f"No Cα coordinates found for {pdb_id}_{chain_id}")
                return None

            coordinates = np.array(ca_coords)

            # Cache in memory for this session
            self._coordinate_cache[cache_key] = coordinates

            return coordinates

        except Exception as e:
            logger.error(f"Failed to extract coordinates for {pdb_id}_{chain_id}: {e}")
            return None

    def check_template_available(self, pdb_id: str, chain_id: str) -> bool:
        """
        Check if a template is available for coordinate extraction.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            True if template is available, False otherwise
        """
        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"
        return pdb_file.exists()

    def get_cache_size_gb(self) -> float:
        """Get current cache size in GB."""
        total_size = 0
        for file_path in self.cache_dir.glob("*.pdb"):
            total_size += file_path.stat().st_size
        return total_size / (1024**3)

    def get_statistics(self) -> Dict[str, Any]:
        """Get template manager statistics."""
        pdb_files = list(self.cache_dir.glob("*.pdb"))

        stats = {
            'total_templates': len(pdb_files),
            'cache_size_gb': self.get_cache_size_gb(),
            'cache_directory': str(self.cache_dir),
            'in_memory_cache_size': len(self._coordinate_cache)
        }

        # Calculate average file size
        if pdb_files:
            total_size = sum(f.stat().st_size for f in pdb_files)
            stats['avg_file_size_mb'] = (total_size / len(pdb_files)) / (1024**2)

        return stats

    def clear_in_memory_cache(self):
        """Clear the in-memory coordinate cache."""
        self._coordinate_cache.clear()
        logger.info("Cleared in-memory coordinate cache")

    def populate_from_homology_results(self, homology_results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Download templates based on homology search results.

        Args:
            homology_results: Dictionary containing homology search results

        Returns:
            Dictionary mapping PDB IDs to download success status
        """
        download_results = {}
        unique_pdbs = set()

        # Collect unique PDB IDs from results
        for query_id, query_data in homology_results.items():
            structural_templates = query_data.get('structural_templates', [])
            for template in structural_templates:
                pdb_id = template['pdb_id']
                unique_pdbs.add(pdb_id)

        logger.info(f"Downloading {len(unique_pdbs)} unique templates...")

        # Download templates
        for pdb_id in tqdm(unique_pdbs, desc="Downloading templates"):
            success = self.download_template(pdb_id) is not None
            download_results[pdb_id] = success

        successful = sum(1 for success in download_results.values() if success)
        logger.info(f"Successfully downloaded {successful}/{len(unique_pdbs)} templates")

        return download_results