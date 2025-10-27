#!/usr/bin/env python3
"""
Simplified dataset download script using gdown.

This script downloads and extracts a dataset from Google Drive using the gdown library,
which handles Google Drive's authentication and confirmation flow automatically.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import gdown
import yaml
from tqdm import tqdm


def load_config():
    """Load download links from YAML configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config or 'datasets' not in config:
            print("Error: Invalid configuration file format")
            sys.exit(1)

        # Get the ESM2 contact prediction dataset link
        datasets = config['datasets']
        if 'esm2_contact_prediction' not in datasets:
            print("Error: esm2_contact_prediction dataset not found in configuration")
            sys.exit(1)

        dataset_config = datasets['esm2_contact_prediction']
        if 'url' not in dataset_config:
            print("Error: URL not found for esm2_contact_prediction dataset")
            sys.exit(1)

        download_link = dataset_config['url']
        dataset_name = dataset_config.get('name', 'ESM2 Contact Prediction Dataset')
        description = dataset_config.get('description', 'Dataset for ESM2 protein contact prediction training')

        print(f"📋 Dataset: {dataset_name}")
        print(f"📝 Description: {description}")

        return download_link

    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)


def download_file(url: str, destination: Path) -> bool:
    """
    Download file from Google Drive using gdown.

    Args:
        url: Google Drive URL
        destination: Path to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📥 Downloading from Google Drive...")
        print(f"📁 Destination: {destination}")

        # Extract file ID from URL for better gdown handling
        file_id = None
        if "drive.google.com" in url or "drive.usercontent.google.com" in url:
            # Try to extract file ID from URL parameters
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            file_id = params.get('id', [None])[0]

        if file_id:
            print(f"🔍 Using file ID: {file_id}")
            # Use gdown with file ID for better reliability
            gdown.download(id=file_id, output=str(destination), quiet=False, fuzzy=True)
        else:
            # Fallback to using the URL directly
            gdown.download(url, str(destination), quiet=False, fuzzy=True)

        if destination.exists() and destination.stat().st_size > 1024:  # At least 1KB
            file_size_mb = destination.stat().st_size / (1024 * 1024)
            print(f"✓ Successfully downloaded to {destination}")
            print(f"📊 File size: {file_size_mb:.1f} MB")
            return True
        else:
            print("✗ Download failed - file is too small or doesn't exist")
            if destination.exists():
                print("📄 Downloaded content appears to be HTML, not the actual file")
                destination.unlink()  # Remove the HTML file
            return False

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract zip file to specified directory.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to

    Returns:
        True if successful, False otherwise
    """
    try:
        if not zip_path.exists():
            print(f"✗ Error: Zip file does not exist: {zip_path}")
            return False

        file_size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"📦 Extracting {zip_path.name} ({file_size_mb:.1f} MB) to {extract_to}...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test zip file integrity
            bad_file = zip_ref.testzip()
            if bad_file:
                print(f"✗ Error: Zip file is corrupted (bad file: {bad_file})")
                return False

            # Get file count
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            if total_files == 0:
                print("✗ Warning: Zip file contains no files")
                return False

            print(f"📄 Found {total_files} files in archive")

            # Extract with progress tracking
            for member in tqdm(file_list, desc="Extracting files", unit="files"):
                try:
                    zip_ref.extract(member, extract_to)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to extract {member}: {e}")
                    continue

        print(f"✓ Successfully extracted to {extract_to}")
        return True

    except zipfile.BadZipFile:
        print(f"✗ Error: {zip_path.name} is not a valid zip file")
        return False
    except Exception as e:
        print(f"✗ Error extracting zip file: {e}")
        return False


def verify_extraction(extract_dir: Path) -> bool:
    """
    Verify that extraction was successful.

    Args:
        extract_dir: Directory where files were extracted

    Returns:
        True if files exist, False otherwise
    """
    try:
        files = list(extract_dir.rglob("*"))
        if not files:
            print(f"✗ No files found in {extract_dir}")
            return False

        file_count = len([f for f in files if f.is_file()])
        dir_count = len([f for f in files if f.is_dir()])

        print(f"✓ Extraction verified: {file_count} files and {dir_count} directories")
        return True

    except Exception as e:
        print(f"✗ Error verifying extraction: {e}")
        return False


def main():
    """Main function to orchestrate download and extraction."""
    parser = argparse.ArgumentParser(description="Download and extract dataset from Google Drive")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--keep-zip", action="store_true", help="Keep zip file after extraction")
    parser.add_argument("--data-dir", default="data", help="Data directory name (default: data)")

    args = parser.parse_args()

    # Load download link from YAML configuration
    download_link = load_config()

    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    zip_path = data_dir / "dataset.zip"

    # Check if dataset already exists
    if not args.force and data_dir.exists() and any(data_dir.iterdir()):
        print(f"Data directory {data_dir} already contains files.")
        print("Use --force to re-download or remove the directory first.")
        return

    print(f"Downloading dataset from Google Drive...")
    print(f"Destination: {data_dir}")

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download file
    if not download_file(download_link, zip_path):
        sys.exit(1)

    # Extract zip file
    if not extract_zip(zip_path, data_dir):
        sys.exit(1)

    # Verify extraction
    if not verify_extraction(data_dir):
        sys.exit(1)

    # Clean up zip file unless requested to keep it
    if not args.keep_zip:
        zip_path.unlink()
        print(f"✓ Removed zip file: {zip_path.name}")

    print(f"\n🎉 Dataset successfully downloaded and extracted to {data_dir}")


if __name__ == "__main__":
    main()