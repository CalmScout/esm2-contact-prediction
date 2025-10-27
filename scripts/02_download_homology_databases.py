#!/usr/bin/env python3
"""
Homology Database Downloader

This script downloads HHblits homology search databases from official sources.
It reads configuration from config.yaml and handles downloading, extracting,
and validation of the databases.

Usage:
    python scripts/02_download_homology_databases.py [--db pdb70|uniref30|all] [--base-path PATH]
"""

import os
import sys
import tarfile
import requests
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urlparse
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatabaseDownloader:
    """Download and manage HHblits homology databases."""

    def __init__(self, config_path: str = "config.yaml", keep_archives: bool = False):
        """
        Initialize the downloader with configuration.

        Args:
            config_path: Path to the config.yaml file
            keep_archives: Whether to keep archive files after extraction
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.keep_archives = keep_archives
        # Get homology_databases path from new paths config, fallback to old homology_databases.base_path, then default
        self.base_path = Path(
            self.config.get('paths', {}).get('homology_databases',
            self.config.get('homology_databases', {}).get('base_path', 'data/homology_databases'))
        )

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _validate_database_config(self, db_name: str, db_config: Dict) -> bool:
        """Validate and normalize database configuration values."""
        try:
            # Validate size values and ensure they're numeric
            if 'compressed_size_gb' in db_config:
                value = db_config['compressed_size_gb']
                if isinstance(value, str):
                    logger.warning(f"Compressed size for '{db_name}' is a string: '{value}'. Converting to float.")
                    db_config['compressed_size_gb'] = float(value)
                elif not isinstance(value, (int, float)):
                    logger.warning(f"Invalid type for compressed size in '{db_name}': {type(value)}. Using default 0.")
                    db_config['compressed_size_gb'] = 0

            if 'extracted_size_gb' in db_config:
                value = db_config['extracted_size_gb']
                if isinstance(value, str):
                    logger.warning(f"Extracted size for '{db_name}' is a string: '{value}'. Converting to float.")
                    db_config['extracted_size_gb'] = float(value)
                elif not isinstance(value, (int, float)):
                    logger.warning(f"Invalid type for extracted size in '{db_name}': {type(value)}. Using default 0.")
                    db_config['extracted_size_gb'] = 0

            # Validate required fields
            required_fields = ['name', 'filename', 'url', 'extract_to']
            for field in required_fields:
                if field not in db_config:
                    logger.error(f"Missing required field '{field}' for database '{db_name}'")
                    return False

            return True

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to validate configuration for '{db_name}': {e}")
            return False

    def _check_disk_space(self, required_gb: float) -> bool:
        """Check if enough disk space is available."""
        try:
            stat = os.statvfs(self.base_path)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

            logger.info(f"Available disk space: {available_gb:.1f} GB")
            logger.info(f"Required space: {required_gb:.1f} GB")

            if available_gb < required_gb:
                logger.error(f"Insufficient disk space. Need {required_gb:.1f} GB, have {available_gb:.1f} GB")
                return False

            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Continue anyway

    def _download_with_progress(self, url: str, file_path: Path, expected_size_gb: float = None) -> bool:
        """Download file with progress bar and resume capability."""
        try:
            # Check if file already exists and get its size for resume
            resume_pos = 0
            if file_path.exists():
                resume_pos = file_path.stat().st_size
                logger.info(f"Resuming download from {resume_pos / (1024**2):.1f} MB")

            headers = {}
            if resume_pos > 0:
                headers['Range'] = f'bytes={resume_pos}-'

            response = requests.get(url, headers=headers, stream=True, timeout=30)

            # Check if server supports resume
            if resume_pos > 0 and response.status_code == 206:
                logger.info("Server supports resume, continuing download")
            elif resume_pos > 0 and response.status_code != 206:
                logger.warning("Server doesn't support resume, starting fresh download")
                resume_pos = 0
                file_path.unlink(missing_ok=True)
                response = requests.get(url, stream=True, timeout=30)

            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            if resume_pos > 0:
                total_size += resume_pos

            mode = 'ab' if resume_pos > 0 else 'wb'

            with open(file_path, mode) as f, tqdm(
                desc=file_path.name,
                initial=resume_pos,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            # Verify file size if we have expected size
            if expected_size_gb:
                try:
                    expected_size = float(expected_size_gb)
                    actual_size_gb = file_path.stat().st_size / (1024**3)
                    min_expected_gb = expected_size * 0.95  # Allow 5% tolerance
                    max_expected_gb = expected_size * 1.05  # Allow 5% tolerance

                    if not (min_expected_gb <= actual_size_gb <= max_expected_gb):
                        logger.warning(f"Downloaded file size {actual_size_gb:.1f} GB differs from expected {expected_size:.1f} GB")
                    else:
                        logger.info(f"Downloaded file size verified: {actual_size_gb:.1f} GB")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid expected size value, skipping size validation: {e}")

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            return False
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return False

    def _extract_database(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract the database archive."""
        try:
            logger.info(f"Extracting {archive_path} to {extract_to}")

            # Create extraction directory
            extract_to.mkdir(parents=True, exist_ok=True)

            with tarfile.open(archive_path, 'r:gz') as tar:
                # Get total number of files for progress tracking
                members = [m for m in tar.getmembers() if m.isreg()]

                with tqdm(total=len(members), desc="Extracting") as progress_bar:
                    for member in members:
                        tar.extract(member, path=extract_to)
                        progress_bar.update(1)

            logger.info(f"Successfully extracted to {extract_to}")
            return True

        except tarfile.TarError as e:
            logger.error(f"Extraction failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
            return False

    def _validate_extraction(self, db_config: Dict, extract_path: Path) -> bool:
        """Validate that the database was extracted correctly."""
        try:
            database_files = db_config.get('database_files', [])

            for required_file in database_files:
                file_path = extract_path / required_file
                if not file_path.exists():
                    logger.error(f"Required database file missing: {file_path}")
                    return False

                # Check file is not empty
                if file_path.stat().st_size == 0:
                    logger.error(f"Database file is empty: {file_path}")
                    return False

            logger.info("Database extraction validated successfully")
            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def download_database(self, db_name: str, base_path_override: Optional[Path] = None) -> bool:
        """
        Download a specific database.

        Args:
            db_name: Name of the database ('pdb70' or 'uniref30')
            base_path_override: Override the base path from config

        Returns:
            True if successful, False otherwise
        """
        if base_path_override:
            self.base_path = base_path_override

        # Get database configuration
        db_config = self.config.get('homology_databases', {}).get(db_name)
        if not db_config:
            logger.error(f"Database '{db_name}' not found in configuration")
            return False

        # Validate and normalize database configuration
        if not self._validate_database_config(db_name, db_config):
            logger.error(f"Database configuration validation failed for '{db_name}'")
            return False

        # Setup paths
        self.base_path.mkdir(parents=True, exist_ok=True)
        archive_path = self.base_path / db_config['filename']
        extract_path = self.base_path / db_config['extract_to']

        logger.info(f"Processing database: {db_config['name']}")
        logger.info(f"Base path: {self.base_path}")

        # Check if database is already extracted
        if extract_path.exists() and self._validate_extraction(db_config, extract_path):
            logger.info(f"Database '{db_name}' already exists and is valid. Skipping.")
            return True

        # Check disk space
        try:
            # Get values with detailed logging for debugging
            compressed_raw = db_config.get('compressed_size_gb', 0)
            extracted_raw = db_config.get('extracted_size_gb', 0)

            logger.debug(f"Raw config values - compressed: {compressed_raw} (type: {type(compressed_raw)}), "
                        f"extracted: {extracted_raw} (type: {type(extracted_raw)})")

            # Convert to float with explicit type checking
            compressed_size = float(compressed_raw) if compressed_raw is not None else 0.0
            extracted_size = float(extracted_raw) if extracted_raw is not None else 0.0

            logger.debug(f"Converted values - compressed: {compressed_size} GB, extracted: {extracted_size} GB")

            required_space = compressed_size + extracted_size
            logger.info(f"Required disk space for '{db_name}': {required_space:.1f} GB "
                       f"({compressed_size:.1f} GB compressed + {extracted_size:.1f} GB extracted)")

        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse size values from config for '{db_name}': {e}")
            logger.error(f"Raw values - compressed: {db_config.get('compressed_size_gb')} "
                        f"(type: {type(db_config.get('compressed_size_gb'))}), "
                        f"extracted: {db_config.get('extracted_size_gb')} "
                        f"(type: {type(db_config.get('extracted_size_gb'))})")
            logger.warning("Using conservative default of 0 GB for required space")
            required_space = 0  # Conservative default

        if not self._check_disk_space(required_space):
            return False

        # Download if needed
        if not archive_path.exists():
            url = db_config.get('url')
            if not url:
                logger.error(f"No download URL configured for '{db_name}'")
                return False

            logger.info(f"Downloading from: {url}")
            expected_size = db_config.get('compressed_size_gb')

            if not self._download_with_progress(url, archive_path, expected_size):
                logger.error(f"Failed to download '{db_name}' from: {url}")
                logger.error("Please check the URL and your internet connection")
                return False
        else:
            logger.info(f"Archive already exists: {archive_path}")

        # Extract database
        if not self._extract_database(archive_path, extract_path):
            logger.error(f"Failed to extract '{db_name}'")
            return False

        # Validate extraction
        if not self._validate_extraction(db_config, extract_path):
            logger.error(f"Validation failed for '{db_name}'")
            return False

        # Clean up archive file unless requested to keep it
        if not self.keep_archives and archive_path.exists():
            try:
                archive_size_gb = archive_path.stat().st_size / (1024**3)
                archive_path.unlink()
                logger.info(f"✓ Deleted archive file: {archive_path.name} ({archive_size_gb:.1f} GB) - freed up disk space")
            except Exception as e:
                logger.warning(f"Failed to delete archive file {archive_path}: {e}")

        logger.info(f"Successfully processed database '{db_name}'")
        return True

    def download_all(self, base_path_override: Optional[Path] = None) -> bool:
        """Download all configured databases."""
        databases = self.config.get('homology_databases', {})

        success = True
        for db_name in databases.keys():
            if db_name == 'base_path':
                continue

            logger.info(f"Downloading database: {db_name}")
            if not self.download_database(db_name, base_path_override):
                success = False
                logger.error(f"Failed to download database: {db_name}")

        # Report overall archive cleanup status
        if not self.keep_archives and success:
            logger.info("All archives have been deleted after successful extraction to save disk space")
            logger.info("Use --keep-archives flag if you want to preserve the archive files")

        return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download HHblits homology databases')
    parser.add_argument('--db', choices=['pdb70', 'uniref30', 'all'], default='all',
                       help='Database to download (default: all)')
    parser.add_argument('--base-path', type=str, help='Override base path from config')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--keep-archives', action='store_true',
                       help='Keep archive files after extraction (default: delete to save space)')

    args = parser.parse_args()

    # Initialize downloader
    downloader = DatabaseDownloader(args.config, args.keep_archives)

    # Override base path if provided
    base_path_override = None
    if args.base_path:
        base_path_override = Path(args.base_path)

    # Download databases
    if args.db == 'all':
        success = downloader.download_all(base_path_override)
    else:
        success = downloader.download_database(args.db, base_path_override)

    if success:
        logger.info("All downloads completed successfully!")
        sys.exit(0)
    else:
        logger.error("Some downloads failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()