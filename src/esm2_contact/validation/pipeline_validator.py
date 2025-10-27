"""
Pipeline-level validation for the complete ESM2 contact prediction workflow.

This module provides high-level validation that checks the integrity
and consistency of the entire pipeline from ground truth to CNN dataset.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineValidator:
    """
    Comprehensive pipeline validator for the ESM2 contact prediction workflow.

    Validates consistency between pipeline steps and ensures data integrity
    throughout the entire workflow.
    """

    def __init__(self, data_dir: str = "data/tiny_10"):
        """
        Initialize the pipeline validator.

        Args:
            data_dir: Directory containing pipeline data files
        """
        self.data_dir = Path(data_dir)
        self.validation_results = {
            'ground_truth': {},
            'homology_search': {},
            'cnn_dataset': {},
            'pipeline_integrity': {}
        }

    def validate_ground_truth_creation(self, ground_truth_file: Optional[str] = None) -> Dict:
        """
        Validate ground truth dataset creation.

        Args:
            ground_truth_file: Path to ground truth HDF5 file

        Returns:
            Validation results dictionary
        """
        logger.info("Validating ground truth dataset creation...")

        if ground_truth_file is None:
            ground_truth_file = self.data_dir / "ground_truth_contacts.h5"

        results = {
            'file_exists': False,
            'protein_count': 0,
            'contact_density_stats': {},
            'length_stats': {},
            'data_integrity': {},
            'validation_passed': False
        }

        try:
            # Check file existence
            if not Path(ground_truth_file).exists():
                results['errors'] = ["Ground truth file not found"]
                return results

            results['file_exists'] = True

            # Load sequences from JSON file (our fast ground truth approach)
            sequences_file = self.data_dir / "sequences.json"
            if not sequences_file.exists():
                results['errors'] = ["Sequences file not found"]
                return results

            with open(sequences_file, 'r') as f:
                sequences = json.load(f)

            # Analyze ground truth data
            with h5py.File(ground_truth_file, 'r') as f:
                ground_truth_group = f['contacts']  # Different group name in our version
                protein_ids = list(ground_truth_group.keys())
                results['protein_count'] = len(protein_ids)

                if len(protein_ids) == 0:
                    results['errors'] = ["No proteins found in ground truth data"]
                    return results

                # Analyze contact densities and lengths
                contact_densities = []
                lengths = []

                for protein_id in protein_ids:
                    protein_group = ground_truth_group[protein_id]
                    contact_map = protein_group['contact_map'][:]
                    sequence = sequences.get(protein_id, '')  # Get from JSON file

                    L = len(sequence)
                    lengths.append(L)

                    # Calculate contact density
                    contacts = np.sum(contact_map > 0.5)  # Binary contacts
                    max_possible = L * (L - 1) // 2
                    density = contacts / max_possible if max_possible > 0 else 0
                    contact_densities.append(density)

                # Compute statistics
                results['contact_density_stats'] = {
                    'mean': float(np.mean(contact_densities)),
                    'median': float(np.median(contact_densities)),
                    'min': float(np.min(contact_densities)),
                    'max': float(np.max(contact_densities)),
                    'std': float(np.std(contact_densities))
                }

                results['length_stats'] = {
                    'mean': float(np.mean(lengths)),
                    'median': float(np.median(lengths)),
                    'min': int(np.min(lengths)),
                    'max': int(np.max(lengths)),
                    'std': float(np.std(lengths))
                }

                # Data integrity checks
                results['data_integrity'] = {
                    'all_sequences_valid': all(len(sequences.get(protein_id, '')) > 0
                                             for protein_id in protein_ids),
                    'all_contact_maps_binary': all(np.all(np.isin(ground_truth_group[protein_id]['contact_map'][:], [0, 1]))
                                                   for protein_id in protein_ids),
                    'all_contact_maps_symmetric': all(np.allclose(ground_truth_group[protein_id]['contact_map'][:],
                                                                  ground_truth_group[protein_id]['contact_map'][:].T)
                                                      for protein_id in protein_ids)
                }

                # Validation criteria
                density_mean = results['contact_density_stats']['mean']
                length_range = results['length_stats']['max'] - results['length_stats']['min']

                validation_checks = [
                    density_mean >= 0.01 and density_mean <= 0.1,  # Realistic density range
                    length_range >= 50,  # Diverse lengths
                    results['data_integrity']['all_sequences_valid'],
                    results['data_integrity']['all_contact_maps_binary'],
                    results['data_integrity']['all_contact_maps_symmetric']
                ]

                results['validation_passed'] = all(validation_checks)

                if results['validation_passed']:
                    logger.info(f"✅ Ground truth validation passed ({len(protein_ids)} proteins)")
                else:
                    logger.warning(f"⚠️ Ground truth validation failed")

        except Exception as e:
            results['errors'] = [f"Validation error: {str(e)}"]
            logger.error(f"Ground truth validation error: {e}")

        self.validation_results['ground_truth'] = results
        return results

    def validate_homology_search(self, homology_file: Optional[str] = None) -> Dict:
        """
        Validate homology search results.

        Args:
            homology_file: Path to homology results JSON file

        Returns:
            Validation results dictionary
        """
        logger.info("Validating homology search results...")

        if homology_file is None:
            homology_file = self.data_dir / "homology_results.json"

        results = {
            'file_exists': False,
            'template_stats': {},
            'success_rate': 0.0,
            'template_quality_stats': {},
            'validation_passed': False
        }

        try:
            # Check file existence
            if not Path(homology_file).exists():
                results['errors'] = ["Homology results file not found"]
                return results

            results['file_exists'] = True

            # Load and analyze homology results
            with open(homology_file, 'r') as f:
                homology_data = json.load(f)

            metadata = homology_data['metadata']
            results_data = homology_data['results']

            total_proteins = len(results_data)
            successful_searches = sum(1 for r in results_data.values()
                                     if r.get('success', False) and r.get('templates_found', 0) > 0)

            results['success_rate'] = successful_searches / total_proteins if total_proteins > 0 else 0

            # Analyze template statistics
            template_counts = []
            identities = []
            coverages = []

            for protein_id, result in results_data.items():
                if result.get('success', False) and result.get('templates_found', 0) > 0:
                    template_counts.append(result['templates_found'])

                    for template in result.get('templates', []):
                        identities.append(template.get('sequence_identity', 0))
                        coverages.append(template.get('alignment_coverage', 0))

            if template_counts:
                results['template_stats'] = {
                    'mean_templates': float(np.mean(template_counts)),
                    'median_templates': float(np.median(template_counts)),
                    'min_templates': int(np.min(template_counts)),
                    'max_templates': int(np.max(template_counts))
                }

            if identities:
                results['template_quality_stats'] = {
                    'mean_identity': float(np.mean(identities)),
                    'median_identity': float(np.median(identities)),
                    'min_identity': float(np.min(identities)),
                    'mean_coverage': float(np.mean(coverages)),
                    'median_coverage': float(np.median(coverages))
                }

            # Validation criteria
            validation_checks = [
                results['success_rate'] >= 0.8,  # At least 80% success
                results['template_stats'].get('mean_templates', 0) >= 10,  # Good template coverage
                results['template_quality_stats'].get('mean_identity', 0) >= 0.25,  # Min identity
                results['template_quality_stats'].get('mean_coverage', 0) >= 0.5   # Min coverage
            ]

            results['validation_passed'] = all(validation_checks)

            if results['validation_passed']:
                logger.info(f"✅ Homology search validation passed ({successful_searches}/{total_proteins} proteins)")
            else:
                logger.warning(f"⚠️ Homology search validation failed")

        except Exception as e:
            results['errors'] = [f"Validation error: {str(e)}"]
            logger.error(f"Homology search validation error: {e}")

        self.validation_results['homology_search'] = results
        return results

    def validate_cnn_dataset(self, cnn_dataset_file: Optional[str] = None) -> Dict:
        """
        Validate CNN dataset generation.

        Args:
            cnn_dataset_file: Path to CNN dataset HDF5 file

        Returns:
            Validation results dictionary
        """
        logger.info("Validating CNN dataset generation...")

        if cnn_dataset_file is None:
            cnn_dataset_file = self.data_dir / "cnn_dataset.h5"

        results = {
            'file_exists': False,
            'protein_count': 0,
            'feature_stats': {},
            'data_integrity': {},
            'validation_passed': False
        }

        try:
            # Check file existence
            if not Path(cnn_dataset_file).exists():
                results['errors'] = ["CNN dataset file not found"]
                return results

            results['file_exists'] = True

            # Analyze CNN dataset
            with h5py.File(cnn_dataset_file, 'r') as f:
                cnn_group = f['cnn_data']
                protein_ids = list(cnn_group.keys())
                results['protein_count'] = len(protein_ids)

                if len(protein_ids) == 0:
                    results['errors'] = ["No proteins found in CNN dataset"]
                    return results

                # Analyze features and integrity
                contact_densities = []
                template_counts = []
                channel_shapes = []

                for protein_id in protein_ids:
                    protein_group = cnn_group[protein_id]

                    # Get contact map and calculate density
                    contact_map = protein_group['consensus_contact_map'][:]
                    L = contact_map.shape[0]
                    contacts = np.sum(contact_map > 0.01)
                    density = contacts / (L * (L - 1) / 2)
                    contact_densities.append(density)

                    # Get metadata
                    template_counts.append(protein_group.attrs.get('templates_used', 0))

                    # Get multi-channel input
                    multi_channel = protein_group['multi_channel_input'][:]
                    channel_shapes.append(multi_channel.shape)

                # Compute statistics
                results['feature_stats'] = {
                    'mean_contact_density': float(np.mean(contact_densities)),
                    'mean_templates_used': float(np.mean(template_counts)),
                    'input_channels': int(channel_shapes[0][0]) if channel_shapes else 0
                }

                # Data integrity checks
                results['data_integrity'] = {
                    'all_shapes_consistent': len(set(shape for shape in channel_shapes)) <= 1,
                    'channel_count_correct': results['feature_stats']['input_channels'] == 68,
                    'no_nan_values': all(not np.any(np.isnan(protein_group['multi_channel_input'][:]))
                                       for protein_group in cnn_group.values()),
                    'all_contact_maps_symmetric': all(np.allclose(protein_group['consensus_contact_map'][:],
                                                                protein_group['consensus_contact_map'][:].T)
                                                      for protein_group in cnn_group.values())
                }

                # Validation criteria
                validation_checks = [
                    results['feature_stats']['mean_contact_density'] >= 0.01,  # Some contacts
                    results['feature_stats']['input_channels'] == 68,  # Correct channel count
                    results['data_integrity']['all_shapes_consistent'],
                    results['data_integrity']['channel_count_correct'],
                    results['data_integrity']['no_nan_values'],
                    results['data_integrity']['all_contact_maps_symmetric']
                ]

                results['validation_passed'] = all(validation_checks)

                if results['validation_passed']:
                    logger.info(f"✅ CNN dataset validation passed ({len(protein_ids)} proteins)")
                else:
                    logger.warning(f"⚠️ CNN dataset validation failed")

        except Exception as e:
            results['errors'] = [f"Validation error: {str(e)}"]
            logger.error(f"CNN dataset validation error: {e}")

        self.validation_results['cnn_dataset'] = results
        return results

    def validate_pipeline_integrity(self) -> Dict:
        """
        Validate end-to-end pipeline integrity and consistency.

        Returns:
            Validation results dictionary
        """
        logger.info("Validating pipeline integrity...")

        results = {
            'file_consistency': {},
            'data_consistency': {},
            'processing_chain': {},
            'validation_passed': False
        }

        try:
            # Check file consistency
            required_files = [
                self.data_dir / "sequences.json",
                self.data_dir / "ground_truth.h5",
                self.data_dir / "homology_results.json",
                self.data_dir / "cnn_dataset.h5"
            ]

            results['file_consistency'] = {
                'all_files_exist': all(Path(f).exists() for f in required_files),
                'missing_files': [str(f) for f in required_files if not Path(f).exists()]
            }

            # Check data consistency between steps
            if results['file_consistency']['all_files_exist']:
                # Load sequences and compare counts
                with open(self.data_dir / "sequences.json", 'r') as f:
                    sequences = json.load(f)

                with h5py.File(self.data_dir / "ground_truth.h5", 'r') as f:
                    ground_truth_count = len(f['ground_truth'])

                with h5py.File(self.data_dir / "cnn_dataset.h5", 'r') as f:
                    cnn_dataset_count = len(f['cnn_data'])

                results['data_consistency'] = {
                    'sequence_count': len(sequences),
                    'ground_truth_count': ground_truth_count,
                    'cnn_dataset_count': cnn_dataset_count,
                    'counts_match': len(sequences) == ground_truth_count == cnn_dataset_count
                }

                # Check processing chain
                ground_truth_validation = self.validation_results.get('ground_truth', {})
                homology_validation = self.validation_results.get('homology_search', {})
                cnn_validation = self.validation_results.get('cnn_dataset', {})

                results['processing_chain'] = {
                    'ground_truth_valid': ground_truth_validation.get('validation_passed', False),
                    'homology_search_valid': homology_validation.get('validation_passed', False),
                    'cnn_dataset_valid': cnn_validation.get('validation_passed', False)
                }

            # Overall validation
            validation_checks = [
                results['file_consistency']['all_files_exist'],
                results['data_consistency'].get('counts_match', False),
                results['processing_chain'].get('ground_truth_valid', False),
                results['processing_chain'].get('homology_search_valid', False),
                results['processing_chain'].get('cnn_dataset_valid', False)
            ]

            results['validation_passed'] = all(validation_checks)

            if results['validation_passed']:
                logger.info("✅ Pipeline integrity validation passed")
            else:
                logger.warning("⚠️ Pipeline integrity validation failed")

        except Exception as e:
            results['errors'] = [f"Validation error: {str(e)}"]
            logger.error(f"Pipeline integrity validation error: {e}")

        self.validation_results['pipeline_integrity'] = results
        return results

    def generate_comprehensive_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive validation report for the entire pipeline.

        Args:
            output_file: Path to save the report (optional)

        Returns:
            Complete validation report dictionary
        """
        logger.info("Generating comprehensive pipeline validation report...")

        # Run all validations
        self.validate_ground_truth_creation()
        self.validate_homology_search()
        self.validate_cnn_dataset()
        self.validate_pipeline_integrity()

        # Generate comprehensive report
        report = {
            'validation_timestamp': str(Path.cwd()),
            'pipeline_version': 'tiny_10',
            'overall_status': 'PASSED' if all([
                self.validation_results['ground_truth'].get('validation_passed', False),
                self.validation_results['homology_search'].get('validation_passed', False),
                self.validation_results['cnn_dataset'].get('validation_passed', False),
                self.validation_results['pipeline_integrity'].get('validation_passed', False)
            ]) else 'FAILED',
            'summary': {
                'ground_truth_proteins': self.validation_results['ground_truth'].get('protein_count', 0),
                'homology_success_rate': self.validation_results['homology_search'].get('success_rate', 0),
                'cnn_dataset_proteins': self.validation_results['cnn_dataset'].get('protein_count', 0),
                'mean_contact_density': self.validation_results['ground_truth'].get('contact_density_stats', {}).get('mean', 0),
                'mean_templates_found': self.validation_results['homology_search'].get('template_stats', {}).get('mean_templates', 0)
            },
            'detailed_results': self.validation_results
        }

        # Save report if requested
        if output_file:
            report_path = Path(output_file)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Validation report saved to: {report_path}")

        # Print summary
        print(f"\n🎯 Pipeline Validation Summary")
        print(f"=" * 40)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Ground Truth: {self.validation_results['ground_truth'].get('protein_count', 0)} proteins")
        print(f"Homology Search: {self.validation_results['homology_search'].get('success_rate', 0):.1%} success rate")
        print(f"CNN Dataset: {self.validation_results['cnn_dataset'].get('protein_count', 0)} proteins")
        print(f"Mean Contact Density: {report['summary']['mean_contact_density']:.3f}")
        print(f"Mean Templates per Protein: {report['summary']['mean_templates_found']:.1f}")

        return report

    def run_full_validation(self, output_file: Optional[str] = None) -> Dict:
        """
        Run complete pipeline validation and generate report.

        Args:
            output_file: Path to save validation report (optional)

        Returns:
            Complete validation report
        """
        return self.generate_comprehensive_report(output_file)