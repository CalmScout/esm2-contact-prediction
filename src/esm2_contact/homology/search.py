"""
Template search functionality for finding homologous protein structures.

This module implements tools for searching PDB database to find homologous
templates for query protein sequences using various sequence similarity methods.
"""

import subprocess
import tempfile
import os
import json
import numpy as np
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.PDB import PDBList
import requests
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the config.yaml file

    Returns:
        Configuration dictionary
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def get_database_base_dir() -> str:
    """
    Get the database base directory from config.yaml ONLY.

    This function ensures the project is self-contained and reproducible
    by only using the configured database path.

    Returns:
        Base directory path for homology databases
    """
    config = load_config()
    db_config = config.get('homology_databases', {})
    base_path = db_config.get('base_path', 'data/homology_databases')

    # Make relative paths absolute from project root
    if not Path(base_path).is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        base_path = project_root / base_path

    logger.info(f"Using database path: {base_path}")
    return str(base_path)


class DatabaseConfig:
    """Configuration and management for HHblits databases."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = get_database_base_dir()
        self.base_dir = Path(base_dir)
        self.databases = self._detect_databases()

        if not self.databases:
            logger.warning(f"No databases found in {self.base_dir}")
            logger.info("Run: uv run python scripts/02_download_homology_databases.py --db all")
        else:
            logger.info(f"DatabaseConfig initialized with {len(self.databases)} databases")

    def _detect_databases(self) -> Dict[str, Dict]:
        """Auto-detect available HHblits databases using config."""
        databases = {}
        config = load_config()
        db_config = config.get('homology_databases', {})

        # Check for PDB70 using explicit config
        pdb70_config = db_config.get('pdb70', {})
        if 'path' in pdb70_config:
            pdb70_path = Path(pdb70_config['path'])
            db_name = pdb70_config.get('database_name', 'pdb70')
            if self._is_valid_hhblits_db(pdb70_path, db_name):
                databases['pdb70'] = {
                    'path': str(pdb70_path),
                    'database_name': db_name,
                    'type': 'templates',
                    'status': 'ready',
                    'description': pdb70_config.get('description', 'PDB70 structural templates')
                }
            else:
                logger.warning(f"PDB70 database found at {pdb70_path} but files are not valid")
        else:
            # Fallback: Check for PDB70 in standard locations
            pdb70_variants = ["pdb70", "pdb70_from_mmcif"]
            for variant in pdb70_variants:
                pdb70_path = self.base_dir / variant
                if self._is_valid_hhblits_db(pdb70_path, variant):
                    databases['pdb70'] = {
                        'path': str(pdb70_path),
                        'database_name': variant,
                        'type': 'templates',
                        'status': 'ready',
                        'description': 'PDB70 structural templates (70% identity clustering)'
                    }
                    break

        # Check for UniRef30 using explicit config
        uniref30_config = db_config.get('uniref30', {})
        if 'path' in uniref30_config:
            uniref30_path = Path(uniref30_config['path'])
            db_name = uniref30_config.get('database_name', 'UniRef30_2023_02_hhsuite')
            if self._is_valid_hhblits_db(uniref30_path, db_name):
                databases['uniref30'] = {
                    'path': str(uniref30_path),
                    'database_name': db_name,
                    'type': 'homologs',
                    'status': 'ready',
                    'description': uniref30_config.get('description', 'UniRef30 sequence homologs')
                }
            else:
                logger.warning(f"UniRef30 database found at {uniref30_path} but files are not valid")
        else:
            # Fallback: Check for UniRef30 in standard locations
            uniref30_variants = ["uniref30", "UniRef30_2023_02", "UniRef30_2023_02_hhsuite"]
            for variant in uniref30_variants:
                uniref30_path = self.base_dir / variant
                if self._is_valid_hhblits_db(uniref30_path, variant):
                    databases['uniref30'] = {
                        'path': str(uniref30_path),
                        'database_name': variant,
                        'type': 'homologs',
                        'status': 'ready',
                        'description': 'UniRef30 sequence homologs (30% identity clustering)'
                    }
                    break

        return databases

    def _is_valid_hhblits_db(self, db_path: Path, db_name: str) -> bool:
        """Check if a directory contains valid HHblits database files."""
        if not db_path.exists():
            return False

        # Check for required HHblits database files in both direct and nested structures
        required_extensions = ['_a3m.ffdata', '_a3m.ffindex']

        for ext in required_extensions:
            # Try direct structure: db_path/db_name.ext
            direct_file = db_path / f"{db_name}{ext}"
            if direct_file.exists():
                continue

            # Try nested structure: db_path/db_name/db_name.ext
            nested_file = db_path / db_name / f"{db_name}{ext}"
            if nested_file.exists():
                continue

            # If neither structure found, invalid database
            return False

        return True

    def get_database_path(self, database_type: str) -> Optional[str]:
        """Get path for a specific database type."""
        logger.info(f"Getting database path for type: {database_type}")

        if database_type in self.databases:
            db_info = self.databases[database_type]
            logger.info(f"Database info for {database_type}: {db_info}")

            if db_info['status'] == 'ready':
                # Use the path and database name from database detection
                base_path = Path(db_info['path'])
                db_name = db_info.get('database_name', database_type)

                logger.info(f"Base path for {database_type}: {base_path}")
                logger.info(f"Using database name: {db_name}")

                # For HHblits, we need to pass the database prefix (directory + base name)
                # HHblits will append _cs219.ffdata, _a3m.ffdata, etc. to this prefix
                if base_path.exists():
                    # Check if the required database files exist
                    a3m_file = base_path / f"{db_name}_a3m.ffdata"
                    cs219_file = base_path / f"{db_name}_cs219.ffdata"

                    logger.info(f"Checking for database files: {a3m_file}, {cs219_file}")
                    logger.info(f"A3M file exists: {a3m_file.exists()}")
                    logger.info(f"CS219 file exists: {cs219_file.exists()}")

                    if a3m_file.exists():
                        # Return the full database prefix (directory + base name)
                        db_prefix = base_path / db_name
                        logger.info(f"Returning database prefix: {db_prefix}")
                        return str(db_prefix)
                    else:
                        logger.warning(f"Database files not found in {base_path}")
                else:
                    logger.warning(f"Database directory does not exist: {base_path}")

        logger.warning(f"Database {database_type} not found or not ready")
        return None

    def is_ready(self, database_type: str) -> bool:
        """Check if a database is ready for use."""
        return (database_type in self.databases and
                self.databases[database_type]['status'] == 'ready')

    def get_status_summary(self) -> Dict[str, str]:
        """Get summary of database statuses."""
        return {db_type: info['status'] for db_type, info in self.databases.items()}

    def estimate_extraction_time(self, database_type: str) -> Optional[float]:
        """Estimate extraction time in seconds for a database."""
        if database_type not in self.databases:
            return None

        if self.databases[database_type]['status'] != 'downloaded':
            return None

        # Rough estimates based on file sizes
        if database_type == 'pdb70':
            # ~17GB file, ~5-10 minutes on modern system
            return 600  # 10 minutes
        elif database_type == 'uniref30':
            # ~26GB file, ~8-15 minutes on modern system
            return 900  # 15 minutes

        return None


class TemplateSearchResult:
    """Class to store template search results."""

    def __init__(self, pdb_id: str, chain_id: str, sequence_identity: float,
                 coverage: float, alignment_score: float, query_seq: str,
                 template_seq: str, alignment: Tuple[str, str],
                 e_value: float = 1e-3, database_type: str = "unknown"):
        self.pdb_id = pdb_id
        self.chain_id = chain_id
        self.sequence_identity = sequence_identity
        self.coverage = coverage
        self.alignment_score = alignment_score
        self.query_seq = query_seq
        self.template_seq = template_seq
        self.alignment = alignment
        self.e_value = e_value
        self.database_type = database_type

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Convert NumPy types to native Python types for JSON compatibility
        def _convert_to_python_type(value):
            """Convert NumPy types to native Python types."""
            if hasattr(value, 'item'):  # NumPy scalar
                return value.item()
            elif hasattr(value, 'tolist'):  # NumPy array
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            else:
                return value

        return {
            'pdb_id': self.pdb_id,
            'chain_id': self.chain_id,
            'sequence_identity': _convert_to_python_type(self.sequence_identity),
            'coverage': _convert_to_python_type(self.coverage),
            'alignment_score': _convert_to_python_type(self.alignment_score),
            'query_seq': self.query_seq,
            'template_seq': self.template_seq,
            'alignment': self.alignment,
            'e_value': _convert_to_python_type(self.e_value),
            'database_type': self.database_type
        }


class DualSearchResult:
    """Container for dual database search results."""

    def __init__(self):
        self.structural_templates = []  # From PDB70
        self.sequence_homologs = []     # From UniRef30
        self.query_sequence = ""

    def add_template(self, result: TemplateSearchResult):
        """Add a template result to the appropriate category."""
        if result.database_type == "pdb70":
            self.structural_templates.append(result)
        elif result.database_type == "uniref30":
            self.sequence_homologs.append(result)

    def get_all_results(self) -> List[TemplateSearchResult]:
        """Get all results combined."""
        return self.structural_templates + self.sequence_homologs

    def get_structural_templates_only(self) -> List[TemplateSearchResult]:
        """
        Get only PDB structural templates for template processing.

        This method returns only templates from PDB70 database that have
        actual 3D coordinates, filtering out UniRef sequences.
        """
        return self.structural_templates

    def get_templates_for_processing(self, min_templates: int = 1) -> List[TemplateSearchResult]:
        """
        Get templates suitable for template processing.

        Prioritizes PDB structural templates, but if there are insufficient
        PDB templates, will include high-quality sequence homologs.

        Args:
            min_templates: Minimum number of templates to return

        Returns:
            List of templates suitable for processing
        """
        templates = self.structural_templates.copy()

        # If we don't have enough PDB templates, add some UniRef homologs
        if len(templates) < min_templates and self.sequence_homologs:
            needed = min_templates - len(templates)
            # Add best sequence homologs based on sequence identity
            sorted_homologs = sorted(
                self.sequence_homologs,
                key=lambda x: x.sequence_identity,
                reverse=True
            )
            templates.extend(sorted_homologs[:needed])

        return templates

    def get_summary(self) -> Dict:
        """Get summary of search results."""
        return {
            'structural_templates': len(self.structural_templates),
            'sequence_homologs': len(self.sequence_homologs),
            'total_results': len(self.get_all_results()),
            'pdb_available': len(self.structural_templates) > 0,
            'processing_templates': len(self.get_templates_for_processing(1))
        }


class TemplateSearcher:
    """
    Search for homologous protein structures in PDB database.

    This class provides multiple methods for finding templates:
    1. Local HHblits search (requires HHsuite installation)
    2. NCBI BLAST search (internet connection required)
    3. Local sequence database search
    """

    def __init__(self, method: str = "blast",
                 min_identity: float = 0.3,
                 min_coverage: float = 0.5,
                 max_templates: int = 10,
                 cache_dir: Optional[Path] = None,
                 database_dir: Optional[str] = None):
        """
        Initialize template searcher.

        Args:
            method: Search method ('blast', 'hhblits', 'local', 'dual')
            min_identity: Minimum sequence identity threshold
            min_coverage: Minimum sequence coverage threshold
            max_templates: Maximum number of templates to return
            cache_dir: Directory to cache downloaded templates
            database_dir: Directory containing HHblits databases (auto-detected if None)
        """
        self.method = method
        self.min_identity = min_identity
        self.min_coverage = min_coverage
        self.max_templates = max_templates
        self.cache_dir = cache_dir or Path.home() / ".esm2_contact" / "templates"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Database configuration
        self.db_config = DatabaseConfig(database_dir)

        # PDB download manager
        self.pdb_downloader = PDBList()

        # Check method availability
        self._check_method_availability()

    def _check_method_availability(self):
        """Check if the chosen search method is available."""
        if self.method == "hhblits":
            if not self._check_hhblits():
                logger.warning("HHblits not found. Falling back to BLAST.")
                self.method = "blast"
        elif self.method == "local":
            logger.info("Local search mode - requires pre-built sequence database")

    def _check_hhblits(self) -> bool:
        """Check if HHblits is installed and accessible."""
        try:
            result = subprocess.run(['hhblits', '-h'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def search_templates(self, query_sequence: str,
                        query_id: str = "query") -> List[TemplateSearchResult]:
        """
        Search for homologous templates for a query sequence.

        Args:
            query_sequence: Protein sequence to search templates for
            query_id: Identifier for the query sequence

        Returns:
            List of template search results sorted by score
        """
        logger.info(f"Searching templates for {query_id} using {self.method}")

        if self.method == "dual":
            dual_results = self.search_dual_databases(query_sequence, query_id)
            # Combine all results and sort by quality
            all_results = dual_results.get_all_results()
        elif self.method == "blast":
            all_results = self._search_blast(query_sequence, query_id)
        elif self.method == "hhblits":
            all_results = self._search_hhblits(query_sequence, query_id)
        elif self.method == "local":
            all_results = self._search_local(query_sequence, query_id)
        else:
            raise ValueError(f"Unknown search method: {self.method}")

        # Filter and sort results
        filtered_results = self._filter_results(all_results)
        sorted_results = sorted(filtered_results,
                              key=lambda x: x.sequence_identity * x.coverage,
                              reverse=True)

        logger.info(f"Found {len(sorted_results)} templates passing filters")
        return sorted_results[:self.max_templates]

    def search_dual_databases(self, query_sequence: str,
                             query_id: str = "query") -> DualSearchResult:
        """
        Search both PDB70 and UniRef30 databases simultaneously.

        Args:
            query_sequence: Protein sequence to search templates for
            query_id: Identifier for the query sequence

        Returns:
            DualSearchResult containing both structural templates and sequence homologs
        """
        logger.info(f"=== Dual Database Search Started for {query_id} ===")
        logger.info(f"Query sequence length: {len(query_sequence)}")
        results = DualSearchResult()
        results.query_sequence = query_sequence

        # Search PDB70 for structural templates
        if self.db_config.is_ready('pdb70'):
            logger.info("PDB70 database is ready, searching for structural templates...")
            pdb70_db_path = self.db_config.get_database_path('pdb70')
            logger.info(f"PDB70 database path returned: {pdb70_db_path}")

            pdb70_results = self._search_hhblits(
                query_sequence, query_id,
                database_path=pdb70_db_path,
                database_type='pdb70'
            )
            for result in pdb70_results:
                result.database_type = 'pdb70'
                results.add_template(result)
            logger.info(f"PDB70 search completed: {len(pdb70_results)} templates found")
        else:
            logger.warning("PDB70 database not ready, skipping structural template search")

        # Search UniRef30 for sequence homologs
        if self.db_config.is_ready('uniref30'):
            logger.info("UniRef30 database is ready, searching for sequence homologs...")
            uniref30_db_path = self.db_config.get_database_path('uniref30')
            logger.info(f"UniRef30 database path returned: {uniref30_db_path}")

            uniref30_results = self._search_hhblits(
                query_sequence, query_id,
                database_path=uniref30_db_path,
                database_type='uniref30'
            )
            for result in uniref30_results:
                result.database_type = 'uniref30'
                results.add_template(result)
            logger.info(f"UniRef30 search completed: {len(uniref30_results)} homologs found")
        else:
            logger.warning("UniRef30 database not ready, skipping homolog search")

        summary = results.get_summary()
        logger.info(f"=== Dual search completed: {summary['structural_templates']} templates, "
                   f"{summary['sequence_homologs']} homologs ===")

        return results

    def _search_blast(self, query_sequence: str, query_id: str) -> List[TemplateSearchResult]:
        """Search templates using NCBI BLAST."""
        try:
            logger.info("Running NCBI BLAST search...")
            result_handle = NCBIWWW.qblast("blastp", "pdb", query_sequence,
                                         hitlist_size=50, expect=0.001)

            blast_records = list(NCBIXML.parse(result_handle))
            results = []

            for record in blast_records:
                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        # Calculate metrics
                        identity = hsp.identities / hsp.align_length
                        coverage = hsp.align_length / len(query_sequence)

                        # Get PDB ID and chain
                        pdb_info = alignment.hit_id.split('|')
                        pdb_id = pdb_info[1] if len(pdb_info) > 1 else alignment.hit_def[:4]
                        chain_id = alignment.hit_def.split()[-1][0] if len(alignment.hit_def.split()) > 1 else 'A'

                        # Clean PDB ID
                        pdb_id = pdb_id.lower()
                        if len(pdb_id) > 4:
                            pdb_id = pdb_id[:4]

                        if identity >= self.min_identity and coverage >= self.min_coverage:
                            result = TemplateSearchResult(
                                pdb_id=pdb_id,
                                chain_id=chain_id,
                                sequence_identity=identity,
                                coverage=coverage,
                                alignment_score=hsp.score,
                                query_seq=hsp.query,
                                template_seq=hsp.sbjct,
                                alignment=(hsp.query, hsp.sbjct)
                            )
                            results.append(result)

            result_handle.close()
            return results

        except Exception as e:
            logger.error(f"BLAST search failed: {e}")
            return []

    def _search_hhblits(self, query_sequence: str, query_id: str,
                    database_path: Optional[str] = None,
                    database_type: str = "pdb70") -> List[TemplateSearchResult]:
        """Search templates using HHblits."""
        logger.info(f"=== HHblits Search Started ===")
        logger.info(f"Database type: {database_type}")
        logger.info(f"Provided database_path: {database_path}")

        try:
            # Use default database path if not provided
            if database_path is None:
                logger.info(f"Database path is None, getting default path")
                database_path = self._get_default_database_path(database_type)
                logger.info(f"Got default database path: {database_path}")
                if database_path is None:
                    logger.error(f"HHblits database {database_type} not found")
                    return []
            else:
                logger.info(f"Using provided database path: {database_path}")

            # Check if database prefix exists (should include base name)
            db_path_obj = Path(database_path)
            logger.info(f"Database prefix: {database_path}")
            logger.info(f"Database prefix path exists: {db_path_obj.exists()}")

            # For debugging, check the parent directory contents
            if db_path_obj.exists():
                logger.info(f"Database directory contents: {list(db_path_obj.parent.iterdir())}")
            else:
                # Check if parent directory exists
                parent_dir = db_path_obj.parent
                if parent_dir.exists():
                    logger.info(f"Parent directory exists: {parent_dir}")
                    logger.info(f"Parent directory contents: {list(parent_dir.iterdir())}")
                else:
                    logger.warning(f"Parent directory does not exist: {parent_dir}")

            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
                f.write(f">{query_id}\n{query_sequence}\n")
                query_file = f.name
                logger.info(f"Created query file: {query_file}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.hhr', delete=False) as f:
                output_file = f.name
                logger.info(f"Created output file: {output_file}")

            try:
                # Load HHblits parameters from config
                import yaml
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    template_config = config.get('template_search', {})
                    hhblits_params = template_config.get('hhblits_parameters', {})
                    db_params = hhblits_params.get(database_type, hhblits_params.get('pdb70', {}))

                    e_value = db_params.get('e_value', "1e-3")
                    coverage = db_params.get('coverage', "0.4")
                    max_seq = db_params.get('max_sequences', "1000")
                    iterations = db_params.get('iterations', "3")
                except Exception as e:
                    logger.warning(f"Could not load HHblits config, using defaults: {e}")
                    # Fallback to relaxed defaults
                    e_value = "1e-3"
                    coverage = "0.4"
                    max_seq = "1000"
                    iterations = "3"

                # Build HHblits command
                # Use the database prefix directly (should be directory + base name)
                cmd = [
                    'hhblits',
                    '-i', query_file,
                    '-o', output_file,
                    '-d', database_path,  # Use the database prefix directly
                    '-n', iterations,     # Number of iterations from config
                    '-e', e_value,
                    '-cov', coverage,
                    '-maxseq', max_seq,
                    '-cpu', '2',         # Use max 2 CPUs (constraint)
                    '-oa3m', '/dev/null'  # Suppress A3M output
                ]

                logger.info(f"Running HHblits against {database_type} database")
                logger.info(f"Database path being used: {database_path}")
                logger.info(f"Full HHblits command: {' '.join(cmd)}")

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                logger.info(f"HHblits return code: {result.returncode}")
                if result.returncode != 0:
                    logger.error(f"HHblits failed with return code {result.returncode}")
                    logger.error(f"Stderr: {result.stderr}")
                    logger.error(f"Stdout: {result.stdout}")
                else:
                    logger.info(f"HHblits completed successfully")

                if result.returncode == 0:
                    results = self._parse_hhblits_output(output_file, query_sequence)
                    logger.info(f"Found {len(results)} results from HHblits")
                    return results
                else:
                    return []

            finally:
                # Clean up temporary files
                try:
                    os.unlink(query_file)
                    os.unlink(output_file)
                    logger.info("Cleaned up temporary files")
                except OSError:
                    pass  # Files might already be deleted

        except subprocess.TimeoutExpired:
            logger.error("HHblits search timed out (300s)")
            return []
        except Exception as e:
            logger.error(f"HHblits search failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _get_default_database_path(self, database_type: str) -> Optional[str]:
        """Get default database path for a given database type."""
        base_dir = Path(get_database_base_dir())

        # Try different possible database directory names
        possible_names = {
            "pdb70": ["pdb70", "pdb70_from_mmcif"],
            "uniref30": ["uniref30", "UniRef30_2023_02"]
        }

        if database_type not in possible_names:
            logger.error(f"Unknown database type: {database_type}")
            return None

        for name in possible_names[database_type]:
            db_path = base_dir / name
            if db_path.exists():
                # Check for HHblits database files
                a3m_data = db_path / f"{name}_a3m.ffdata"
                a3m_index = db_path / f"{name}_a3m.ffindex"

                if a3m_data.exists() and a3m_index.exists():
                    logger.info(f"Found {database_type} database at {db_path}")
                    return str(db_path)
                else:
                    logger.info(f"Database directory {db_path} exists but missing HHblits format files")

        # If no extracted database found, check if tar.gz exists
        tar_file = base_dir / f"{database_type}_from_mmcif_200401.tar.gz"
        if database_type == "uniref30":
            tar_file = base_dir / "UniRef30_2023_02_hhsuite.tar.gz"

        if tar_file.exists():
            logger.warning(f"Database tar file found at {tar_file} but not extracted. "
                          f"Please extract it before using HHblits.")

        return None

    def _search_local(self, query_sequence: str, query_id: str) -> List[TemplateSearchResult]:
        """Search templates using local sequence database."""
        # This would search against a pre-built local database
        # For now, return empty list
        logger.warning("Local search not implemented yet")
        return []

    def _extract_template_sequences(self, output_file: str) -> Dict[str, str]:
        """
        Extract template sequences from HHblits HHR alignment section.

        Args:
            output_file: Path to HHR output file

        Returns:
            Dictionary mapping template identifiers to their sequences
        """
        template_sequences = {}

        try:
            with open(output_file, 'r') as f:
                content = f.read()

            lines = content.split('\n')

            # Find all template sections and extract sequences
            for i, line in enumerate(lines):
                line = line.strip()

                # Look for template sequence lines in alignment sections
                if line.startswith('T ') and not line.startswith('T Consensus'):
                    parts = line.split()
                    if len(parts) >= 5:
                        template_id = parts[1]
                        sequence = parts[3]

                        # Clean up template ID and sequence
                        if template_id and sequence and sequence != parts[2]:  # sequence != start_pos
                            # Remove gaps from sequence
                            clean_sequence = sequence.replace('-', '')
                            if clean_sequence:  # Only add non-empty sequences
                                template_sequences[template_id] = clean_sequence

            logger.info(f"Extracted sequences for {len(template_sequences)} templates")

        except Exception as e:
            logger.warning(f"Error extracting template sequences: {e}")

        return template_sequences

    def _find_template_sequence(self, hit_name: str, template_sequences: Dict[str, str]) -> str:
        """
        Find template sequence for a given hit name using multiple matching strategies.

        Args:
            hit_name: Template name from HHblits hit table
            template_sequences: Dictionary of template sequences extracted from alignments

        Returns:
            Template sequence string or empty string if not found
        """
        # Try direct match first
        if hit_name in template_sequences:
            return template_sequences[hit_name]

        # Try common variations
        variations = [
            hit_name.lower(),
            hit_name.upper(),
            hit_name.replace('pdb', ''),
            hit_name.replace('PDB', ''),
        ]

        for variation in variations:
            if variation in template_sequences:
                return template_sequences[variation]

        # Try to extract PDB ID and match with sequences
        pdb_id, chain_id = self._parse_pdb_from_hit(hit_name)
        template_key = f"{pdb_id}_{chain_id}"

        if template_key in template_sequences:
            return template_sequences[template_key]

        # Try just PDB ID
        if pdb_id in template_sequences:
            return template_sequences[pdb_id]

        # Try partial matches (for cases where formatting differs)
        for template_key, sequence in template_sequences.items():
            if pdb_id.lower() in template_key.lower() or template_key.lower() in pdb_id.lower():
                return sequence

        return ""

    def _parse_hhblits_output(self, output_file: str, query_sequence: str,
                               min_identity: float = 0.25, min_coverage: float = 0.50) -> List[TemplateSearchResult]:
        """
        Simple and robust HHblits HHR output file parser.

        This parser extracts basic hit information without requiring complex alignment processing,
        which makes it much more reliable and less prone to format changes.
        """
        results = []

        try:
            # First, extract template sequences from the alignment section
            template_sequences = self._extract_template_sequences(output_file)
            logger.info(f"Found {len(template_sequences)} template sequences for matching")

            with open(output_file, 'r') as f:
                content = f.read()

            # Process each line looking for hit data
            # Format: rank name description prob e-value p-value score ss_score cols query_range template_range (length)
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()

                # Skip lines that don't start with a rank number (allowing leading whitespace)
                if not line or not line[0].isdigit():
                    continue

                parts = line.split()
                if len(parts) < 10:
                    logger.debug(f"Skipping line {line_num}: insufficient parts ({len(parts)} < 10): {line[:80]}")
                    continue

                try:
                    # Parse basic info that's always in fixed positions
                    rank = int(parts[0])
                    name = parts[1]

                    # Use a more robust approach to find probability
                    # Look for probability in the expected numeric data section
                    # HHR format: rank name [description...] prob e-value p-value score ss_score cols query_range template_range

                    prob_idx = None
                    # Skip first 2 parts (rank, name), then look for probability
                    # We need at least 8 parts after probability for the parsing to work
                    for i in range(2, min(len(parts) - 6, len(parts))):
                        part = parts[i]
                        try:
                            val = float(part)
                            # Check if this is likely a probability (0-100, typically 50-100 for HHblits)
                            if 0 <= val <= 100:
                                # Additional validation: check if following parts look like valid numeric values
                                try:
                                    # Try to parse next few values to see if they make sense
                                    test_evalue = float(parts[i + 1])
                                    test_pvalue = float(parts[i + 2])
                                    test_score = float(parts[i + 3])
                                    test_ss_score = float(parts[i + 4])
                                    test_cols = int(parts[i + 5])

                                    # Additional validation: check if we have enough remaining parts for ranges
                                    if i + 7 < len(parts):
                                        # If we can parse these successfully, this is likely the probability
                                        prob_idx = i
                                        break
                                except (ValueError, IndexError):
                                    # If next values don't parse as expected, this isn't the probability
                                    # Continue searching for probability, don't break out of the loop
                                    continue
                        except ValueError:
                            continue

                    if prob_idx is None:
                        logger.debug(f"Could not find valid probability in line {line_num}: {line[:80]}...")
                        continue

                    # Extract values based on found probability index
                    probability = float(parts[prob_idx])
                    e_value = float(parts[prob_idx + 1])
                    p_value = float(parts[prob_idx + 2])
                    score = float(parts[prob_idx + 3])
                    ss_score = float(parts[prob_idx + 4])  # Fixed: use float instead of int
                    cols = int(parts[prob_idx + 5])

                    # Extract query and template ranges (remaining parts)
                    query_range = parts[prob_idx + 6]
                    template_range = parts[prob_idx + 7]

                    # Description is everything between name and probability
                    description = ' '.join(parts[2:prob_idx])

                    # Calculate identity from probability
                    identity = probability / 100.0

                    # Estimate coverage from template range information
                    # Parse the template_range to get actual alignment coverage
                    coverage = 0.5  # Default fallback coverage
                    try:
                        # Template range format is typically "start-end"
                        if '-' in template_range:
                            start_str, end_str = template_range.split('-')
                            start_pos = int(start_str)
                            end_pos = int(end_str)
                            aligned_length = end_pos - start_pos + 1
                            # Calculate coverage as aligned positions / total template sequence length
                            coverage = min(1.0, aligned_length / len(query_sequence))
                        # Alternative: use cols as alignment length if available and reasonable
                        elif cols > 10 and cols < len(query_sequence) * 2:  # Reasonable range check
                            coverage = min(1.0, cols / len(query_sequence))
                    except (ValueError, IndexError):
                        # If parsing fails, use a reasonable default based on identity
                        coverage = min(1.0, identity + 0.1)  # Slightly higher than identity

                    # Apply filtering criteria
                    if (identity >= min_identity and
                        coverage >= min_coverage and
                        e_value <= 0.001):  # Standard E-value threshold

                        # Extract PDB ID and chain ID with improved parsing
                        pdb_id, chain_id = self._parse_pdb_from_hit(name)

                        # Get template sequence with better matching
                        template_seq = self._find_template_sequence(name, template_sequences)

                        if not template_seq:
                            logger.debug(f"No template sequence found for {name} (parsed as {pdb_id}_{chain_id})")
                            # Still include the template even if sequence not found
                            template_seq = ""

                        results.append(TemplateSearchResult(
                            pdb_id=pdb_id,
                            chain_id=chain_id,
                            sequence_identity=identity,
                            coverage=coverage,
                            e_value=e_value,
                            alignment_score=score,
                            query_seq=query_sequence,  # Use original sequence
                            template_seq=template_seq,  # Use extracted template sequence
                            alignment=("", ""),  # Empty alignment for simple parser
                            database_type='pdb70' if 'pdb70' in output_file else 'uniref30'
                        ))

                except (ValueError, IndexError, AttributeError) as e:
                    logger.warning(f"Failed to parse hit at line {line_num}: {line[:80]}... Error: {e}")
                    logger.debug(f"Line details - Total parts: {len(parts)}, Parts: {parts[:15]}")  # Show first 15 parts
                    logger.debug(f"Attempted probability index: {prob_idx if 'prob_idx' in locals() else 'not found'}")
                    continue

            logger.info(f"Successfully parsed {len(results)} hits from HHblits output using enhanced parser")
            return results

        except Exception as e:
            logger.error(f"Error in simple HHblits parsing: {e}")
            return results

    def _process_hhblits_hit(self, hit_info: Dict, alignment_lines: List[str],
                           query_sequence: str, query_aligned: str) -> Optional[TemplateSearchResult]:
        """Process a single HHblits hit and calculate alignment metrics."""
        try:
            # Extract template sequence from alignment
            template_aligned = None
            query_start, query_end = 0, 0
            template_start, template_end = 0, 0

            for line in alignment_lines:
                if line.startswith(' ') and 'Consensus' not in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        # This is template alignment line
                        template_aligned = parts[3]

                        # Extract position information if available
                        if len(parts) >= 7:
                            template_start = int(parts[1])
                            template_end = int(parts[2])
                        break

            if not template_aligned:
                logger.warning(f"Could not extract template alignment for hit {hit_info.get('name', 'Unknown')}")
                return None

            # Calculate sequence identity and coverage
            identity = self._calculate_sequence_identity(query_aligned, template_aligned)
            coverage = len(query_aligned.replace('-', '')) / len(query_sequence)

            # Parse PDB ID and chain from hit name
            pdb_id, chain_id = self._parse_pdb_from_hit(hit_info['name'])

            # Create template result
            result = TemplateSearchResult(
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence_identity=identity,
                coverage=coverage,
                alignment_score=hit_info['score'],
                query_seq=query_aligned,
                template_seq=template_aligned,
                alignment=(query_aligned, template_aligned)
            )

            return result

        except Exception as e:
            logger.error(f"Error processing HHblits hit: {e}")
            return None

    def _calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity between two aligned sequences."""
        if len(seq1) != len(seq2):
            return 0.0

        matches = sum(1 for a, b in zip(seq1, seq2)
                     if a != '-' and b != '-' and a == b)
        total_positions = sum(1 for a, b in zip(seq1, seq2)
                            if a != '-' and b != '-')

        return matches / total_positions if total_positions > 0 else 0.0

    def _parse_pdb_from_hit(self, hit_name: str) -> Tuple[str, str]:
        """Parse PDB ID and chain from HHblits hit name."""
        import re

        hit_name = hit_name.strip()

        # Remove common prefixes
        prefixes_to_remove = ['pdb', 'PDB', 'mmCIF', 'mmCIF_']
        for prefix in prefixes_to_remove:
            if hit_name.startswith(prefix):
                hit_name = hit_name[len(prefix):]

        # Handle UniRef30 format (typically just numbers)
        if hit_name.isdigit():
            return hit_name[:8], 'A'  # Return UniRef30 ID as-is

        # Split by common separators
        if '_' in hit_name:
            parts = hit_name.split('_')
            pdb_id = parts[0]
            chain_id = parts[1] if len(parts) > 1 else 'A'
        elif '|' in hit_name:
            parts = hit_name.split('|')
            pdb_id = parts[0]
            chain_id = 'A'
            # Try to extract chain from description
            if len(parts) > 1:
                desc = parts[1].lower()
                if 'chain=' in desc:
                    chain_part = desc.split('chain=')[1]
                    chain_id = chain_part[0] if chain_part else 'A'
                elif 'chain' in desc:
                    # Look for chain info after "chain" keyword
                    chain_match = re.search(r'chain\s*([A-Za-z0-9])', desc)
                    if chain_match:
                        chain_id = chain_match.group(1)
        else:
            # Try to extract PDB ID pattern
            # Look for 4-character alphanumeric patterns
            pdb_match = re.search(r'([a-zA-Z0-9]{4})', hit_name)
            if pdb_match:
                pdb_id = pdb_match.group(1)
                # Try to find chain ID after PDB ID
                remaining = hit_name[pdb_match.end():]
                chain_match = re.search(r'([A-Za-z0-9])', remaining)
                chain_id = chain_match.group(1) if chain_match else 'A'
            else:
                # Fallback: use first 4 chars
                pdb_id = hit_name[:4]
                chain_id = hit_name[4:5] if len(hit_name) > 4 else 'A'

        # Clean up PDB ID
        pdb_id = re.sub(r'[^a-zA-Z0-9]', '', pdb_id).lower()

        # Clean up chain ID
        chain_id = re.sub(r'[^a-zA-Z0-9]', '', chain_id).upper()

        # Ensure PDB ID has reasonable length
        if len(pdb_id) > 4:
            pdb_id = pdb_id[:4]
        elif len(pdb_id) < 4:
            # Pad or use as-is for UniRef format
            if pdb_id.isdigit():
                pass  # Keep as-is for UniRef
            else:
                pdb_id = pdb_id.ljust(4, '0')[:4]

        # Ensure chain ID is not empty
        if not chain_id:
            chain_id = 'A'

        return pdb_id, chain_id

    def _filter_results(self, results: List[TemplateSearchResult]) -> List[TemplateSearchResult]:
        """Filter search results based on quality thresholds."""
        filtered = []
        for result in results:
            if (result.sequence_identity >= self.min_identity and
                result.coverage >= self.min_coverage):
                filtered.append(result)
        return filtered

    def download_template(self, pdb_id: str) -> Optional[Path]:
        """
        Download template PDB file if not already cached.

        Args:
            pdb_id: PDB identifier to download

        Returns:
            Path to downloaded PDB file or None if failed
        """
        pdb_file = self.cache_dir / f"{pdb_id}.pdb"

        if pdb_file.exists():
            logger.debug(f"Template {pdb_id} already cached")
            return pdb_file

        try:
            logger.info(f"Downloading template {pdb_id}")
            self.pdb_downloader.retrieve_pdb_file(pdb_id, pdir=str(self.cache_dir),
                                                file_format='pdb')

            # HHblits downloads with .ent extension, rename to .pdb
            ent_file = self.cache_dir / f"pdb{pdb_id}.ent"
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

    def save_search_results(self, results: List[TemplateSearchResult],
                           output_file: Path):
        """Save search results to JSON file."""
        data = {
            'method': self.method,
            'min_identity': self.min_identity,
            'min_coverage': self.min_coverage,
            'max_templates': self.max_templates,
            'results': [result.to_dict() for result in results]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(results)} search results to {output_file}")

    def load_search_results(self, input_file: Path) -> List[TemplateSearchResult]:
        """Load search results from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)

        results = []
        for result_data in data['results']:
            result = TemplateSearchResult(
                pdb_id=result_data['pdb_id'],
                chain_id=result_data['chain_id'],
                sequence_identity=result_data['sequence_identity'],
                coverage=result_data['coverage'],
                alignment_score=result_data['alignment_score'],
                query_seq=result_data['query_seq'],
                template_seq=result_data['template_seq'],
                alignment=tuple(result_data['alignment'])
            )
            results.append(result)

        logger.info(f"Loaded {len(results)} search results from {input_file}")
        return results