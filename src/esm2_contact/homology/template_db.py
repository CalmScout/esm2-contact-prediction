"""
Template database management for homology-based contact prediction.

This module provides tools for managing a local database of protein templates,
including downloading, caching, and retrieving template structures.
"""

import json
import sqlite3
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import h5py
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBList import PDBList
from Bio import SeqIO
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TemplateInfo:
    """Class to store template metadata."""

    def __init__(self, pdb_id: str, chain_id: str, sequence: str,
                 length: int, resolution: Optional[float] = None,
                 method: Optional[str] = None, title: Optional[str] = None,
                 deposition_date: Optional[str] = None):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.sequence = sequence
        self.length = length
        self.resolution = resolution
        self.method = method
        self.title = title
        self.deposition_date = deposition_date
        self.file_path = None
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

    @classmethod
    def from_dict(cls, data: Dict) -> 'TemplateInfo':
        """Create from dictionary."""
        info = cls(
            pdb_id=data['pdb_id'],
            chain_id=data['chain_id'],
            sequence=data['sequence'],
            length=data['length'],
            resolution=data.get('resolution'),
            method=data.get('method'),
            title=data.get('title'),
            deposition_date=data.get('deposition_date')
        )
        if data.get('file_path'):
            info.file_path = Path(data['file_path'])
        if data.get('last_accessed'):
            info.last_accessed = datetime.fromisoformat(data['last_accessed'])
        return info


class TemplateDatabase:
    """
    Database manager for protein templates.

    This class manages a local database of protein structures for
    template-based modeling, including downloading, caching, and
    efficient retrieval of template information.
    """

    def __init__(self, db_path: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 max_cache_size_gb: float = 10.0):
        """
        Initialize template database.

        Args:
            db_path: Path to SQLite database file
            cache_dir: Directory to cache template files
            max_cache_size_gb: Maximum cache size in GB
        """
        self.db_path = db_path or Path.home() / ".esm2_contact" / "templates.db"
        self.cache_dir = cache_dir or Path.home() / ".esm2_contact" / "template_cache"
        self.max_cache_size_gb = max_cache_size_gb

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        # PDB downloader
        self.pdb_downloader = PDBList()
        self.parser = PDBParser(QUIET=True)

    def _init_database(self):
        """Initialize SQLite database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS templates (
                    pdb_id TEXT,
                    chain_id TEXT,
                    sequence TEXT,
                    length INTEGER,
                    resolution REAL,
                    method TEXT,
                    title TEXT,
                    deposition_date TEXT,
                    file_path TEXT,
                    last_accessed TEXT,
                    file_hash TEXT,
                    PRIMARY KEY (pdb_id, chain_id)
                )
            ''')

            # Create search cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT,
                    query_id TEXT,
                    results TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (query_hash)
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_template_length ON templates (length)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_template_resolution ON templates (resolution)')

            conn.commit()

    def add_template(self, template_info: TemplateInfo,
                     update_if_exists: bool = True) -> bool:
        """
        Add a template to the database.

        Args:
            template_info: Template information
            update_if_exists: Whether to update existing entry

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if update_if_exists:
                    cursor.execute('''
                        INSERT OR REPLACE INTO templates
                        (pdb_id, chain_id, sequence, length, resolution, method,
                         title, deposition_date, file_path, last_accessed, file_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        template_info.pdb_id,
                        template_info.chain_id,
                        template_info.sequence,
                        template_info.length,
                        template_info.resolution,
                        template_info.method,
                        template_info.title,
                        template_info.deposition_date,
                        str(template_info.file_path) if template_info.file_path else None,
                        template_info.last_accessed.isoformat(),
                        self._calculate_file_hash(template_info.file_path) if template_info.file_path else None
                    ))
                else:
                    cursor.execute('''
                        INSERT OR IGNORE INTO templates
                        (pdb_id, chain_id, sequence, length, resolution, method,
                         title, deposition_date, file_path, last_accessed, file_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        template_info.pdb_id,
                        template_info.chain_id,
                        template_info.sequence,
                        template_info.length,
                        template_info.resolution,
                        template_info.method,
                        template_info.title,
                        template_info.deposition_date,
                        str(template_info.file_path) if template_info.file_path else None,
                        template_info.last_accessed.isoformat(),
                        self._calculate_file_hash(template_info.file_path) if template_info.file_path else None
                    ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to add template {template_info.pdb_id}_{template_info.chain_id}: {e}")
            return False

    def get_template(self, pdb_id: str, chain_id: str) -> Optional[TemplateInfo]:
        """
        Get template information from database.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            TemplateInfo object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM templates WHERE pdb_id = ? AND chain_id = ?
                ''', (pdb_id, chain_id))

                row = cursor.fetchone()
                if row:
                    # Update last accessed
                    cursor.execute('''
                        UPDATE templates SET last_accessed = ? WHERE pdb_id = ? AND chain_id = ?
                    ''', (datetime.now().isoformat(), pdb_id, chain_id))
                    conn.commit()

                    return self._row_to_template_info(row)
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to get template {pdb_id}_{chain_id}: {e}")
            return None

    def search_templates(self, query_length: Optional[int] = None,
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        max_resolution: Optional[float] = None,
                        method: Optional[str] = None,
                        limit: int = 100) -> List[TemplateInfo]:
        """
        Search for templates matching criteria.

        Args:
            query_length: Query sequence length (used for similarity)
            min_length: Minimum template length
            max_length: Maximum template length
            max_resolution: Maximum resolution
            method: Experimental method (X-RAY, NMR, etc.)
            limit: Maximum number of results

        Returns:
            List of matching templates
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Build query
                query = "SELECT * FROM templates WHERE 1=1"
                params = []

                if min_length is not None:
                    query += " AND length >= ?"
                    params.append(min_length)

                if max_length is not None:
                    query += " AND length <= ?"
                    params.append(max_length)

                if max_resolution is not None:
                    query += " AND resolution <= ?"
                    params.append(max_resolution)

                if method is not None:
                    query += " AND method LIKE ?"
                    params.append(f"%{method}%")

                # Add ordering by relevance to query length
                if query_length is not None:
                    query += " ORDER BY ABS(length - ?) ASC, resolution ASC"
                    params.append(query_length)
                else:
                    query += " ORDER BY resolution ASC, length ASC"

                query += " LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [self._row_to_template_info(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to search templates: {e}")
            return []

    def download_template(self, pdb_id: str, force_download: bool = False) -> Optional[Path]:
        """
        Download template PDB file.

        Args:
            pdb_id: PDB identifier
            force_download: Force re-download even if file exists

        Returns:
            Path to downloaded file or None if failed
        """
        pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"

        if not force_download and pdb_file.exists():
            logger.debug(f"Template {pdb_id} already cached")
            return pdb_file

        try:
            logger.info(f"Downloading template {pdb_id}")
            self.pdb_downloader.retrieve_pdb_file(
                pdb_id, pdir=str(self.cache_dir), file_format='pdb'
            )

            # Rename .ent file to .pdb
            ent_file = self.cache_dir / f"pdb{pdb_id.lower()}.ent"
            if ent_file.exists():
                ent_file.rename(pdb_file)

            if pdb_file.exists():
                logger.info(f"Successfully downloaded {pdb_id}")
                return pdb_file
            else:
                logger.error(f"Failed to download {pdb_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading template {pdb_id}: {e}")
            return None

    def parse_template(self, pdb_file: Path) -> Dict[str, TemplateInfo]:
        """
        Parse PDB file and extract chain information.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Dictionary mapping chain IDs to TemplateInfo objects
        """
        chains = {}

        try:
            structure = self.parser.get_structure('template', str(pdb_file))

            # Extract metadata from PDB header
            resolution = None
            method = None
            title = None
            deposition_date = None

            if hasattr(structure, 'header'):
                header = structure.header
                resolution = header.get('resolution')
                method = header.get('structure_method')
                title = header.get('name')
                deposition_date = header.get('deposition_date')

            # Parse chains
            for model in structure:
                for chain in model:
                    sequence = ""
                    residues = []

                    for residue in chain:
                        # Skip hetero residues and water
                        if residue.id[0] != ' ':
                            continue

                        # Get residue name
                        res_name = residue.get_resname()
                        if res_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
                                       'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                                       'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                            # Map to one-letter code
                            aa_map = {
                                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
                                'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
                                'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                                'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
                                'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
                            }
                            sequence += aa_map[res_name]
                            residues.append(residue)

                    if len(sequence) >= 10:  # Only keep chains with reasonable length
                        chain_info = TemplateInfo(
                            pdb_id=pdb_file.stem.lower(),
                            chain_id=chain.id,
                            sequence=sequence,
                            length=len(sequence),
                            resolution=resolution,
                            method=method,
                            title=title,
                            deposition_date=deposition_date
                        )
                        chain_info.file_path = pdb_file
                        chains[chain.id] = chain_info

            return chains

        except Exception as e:
            logger.error(f"Failed to parse template {pdb_file}: {e}")
            return {}

    def populate_from_pdb_file(self, pdb_file: Path) -> int:
        """
        Parse and add all chains from a PDB file to the database.

        Args:
            pdb_file: Path to PDB file

        Returns:
            Number of chains added
        """
        chains = self.parse_template(pdb_file)
        added_count = 0

        for chain_id, template_info in chains.items():
            if self.add_template(template_info):
                added_count += 1

        logger.info(f"Added {added_count}/{len(chains)} chains from {pdb_file}")
        return added_count

    def get_template_coordinates(self, pdb_id: str, chain_id: str) -> Optional[np.ndarray]:
        """
        Get Cα coordinates for a template.

        Args:
            pdb_id: PDB identifier
            chain_id: Chain identifier

        Returns:
            Array of Cα coordinates (L, 3) or None if not found
        """
        template_info = self.get_template(pdb_id, chain_id)
        if not template_info:
            return None

        # Use file_path from template info, or construct it from cache
        pdb_file = template_info.file_path
        if not pdb_file:
            pdb_file = self.cache_dir / f"{pdb_id.lower()}.pdb"
            if not pdb_file.exists():
                logger.error(f"PDB file not found for {pdb_id}_{chain_id}")
                return None

        try:
            structure = self.parser.get_structure('template', str(pdb_file))
            ca_coords = []

            # Only use first model
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            if residue.id[0] != ' ':  # Skip hetero residues
                                continue
                            if 'CA' in residue:
                                ca_coords.append(residue['CA'].get_coord())
                        break  # Found the target chain
                break  # Only use first model

            return np.array(ca_coords) if ca_coords else None

        except Exception as e:
            logger.error(f"Failed to get coordinates for {pdb_id}_{chain_id}: {e}")
            return None

    def cache_search_results(self, query_id: str, query_sequence: str,
                           results: List[Dict]):
        """
        Cache search results for future use.

        Args:
            query_id: Query identifier
            query_sequence: Query sequence
            results: Search results to cache
        """
        query_hash = hashlib.md5(query_sequence.encode()).hexdigest()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO search_cache
                    (query_hash, query_id, results, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    query_hash,
                    query_id,
                    json.dumps(results),
                    datetime.now().isoformat()
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")

    def get_cached_search_results(self, query_sequence: str) -> Optional[List[Dict]]:
        """
        Get cached search results for a query sequence.

        Args:
            query_sequence: Query sequence

        Returns:
            Cached results or None if not found
        """
        query_hash = hashlib.md5(query_sequence.encode()).hexdigest()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT results FROM search_cache WHERE query_hash = ?
                ''', (query_hash,))

                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                else:
                    return None

        except Exception as e:
            logger.error(f"Failed to get cached search results: {e}")
            return None

    def cleanup_cache(self):
        """Clean up template cache based on size and access time."""
        try:
            # Calculate total cache size
            total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            max_size_bytes = self.max_cache_size_gb * 1024**3

            if total_size <= max_size_bytes:
                return

            logger.info(f"Cache size ({total_size/1024**3:.2f} GB) exceeds limit ({self.max_cache_size_gb} GB)")

            # Get least recently accessed files
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT file_path FROM templates
                    WHERE file_path IS NOT NULL
                    ORDER BY last_accessed ASC
                ''')

                files_to_remove = []
                current_size = total_size

                for row in cursor.fetchall():
                    if current_size <= max_size_bytes * 0.8:  # Leave 20% buffer
                        break

                    file_path = Path(row[0]) if row[0] else None
                    if file_path and file_path.exists():
                        files_to_remove.append(file_path)
                        current_size -= file_path.stat().st_size

                # Remove files
                for file_path in files_to_remove:
                    file_path.unlink()
                    logger.debug(f"Removed cached file: {file_path}")

                # Update database
                pdb_ids = [(f.stem.lower(),) for f in files_to_remove]
                cursor.executemany('''
                    UPDATE templates SET file_path = NULL WHERE pdb_id = ?
                ''', pdb_ids)

                conn.commit()

            logger.info(f"Cleaned up {len(files_to_remove)} cached files")

        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")

    def _row_to_template_info(self, row) -> TemplateInfo:
        """Convert database row to TemplateInfo object."""
        return TemplateInfo(
            pdb_id=row[0],
            chain_id=row[1],
            sequence=row[2],
            length=row[3],
            resolution=row[4],
            method=row[5],
            title=row[6],
            deposition_date=row[7]
        )

    def _calculate_file_hash(self, file_path: Optional[Path]) -> Optional[str]:
        """Calculate MD5 hash of a file."""
        if not file_path or not file_path.exists():
            return None

        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total templates
                cursor.execute("SELECT COUNT(*) FROM templates")
                total_templates = cursor.fetchone()[0]

                # Chains per PDB
                cursor.execute('''
                    SELECT pdb_id, COUNT(*) FROM templates
                    GROUP BY pdb_id
                    ORDER BY COUNT(*) DESC LIMIT 1
                ''')
                max_chains = cursor.fetchone()
                max_chains_per_pdb = max_chains[1] if max_chains else 0

                # Resolution stats
                cursor.execute("SELECT AVG(resolution), MIN(resolution), MAX(resolution) FROM templates WHERE resolution IS NOT NULL")
                resolution_stats = cursor.fetchone()

                # Length stats
                cursor.execute("SELECT AVG(length), MIN(length), MAX(length) FROM templates")
                length_stats = cursor.fetchone()

                # Cache size
                cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file()) / 1024**3

                return {
                    'total_templates': total_templates,
                    'max_chains_per_pdb': max_chains_per_pdb,
                    'avg_resolution': resolution_stats[0],
                    'min_resolution': resolution_stats[1],
                    'max_resolution': resolution_stats[2],
                    'avg_length': length_stats[0],
                    'min_length': length_stats[1],
                    'max_length': length_stats[2],
                    'cache_size_gb': cache_size,
                    'cache_dir': str(self.cache_dir),
                    'db_path': str(self.db_path)
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}