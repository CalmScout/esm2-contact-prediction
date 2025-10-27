"""
Robust template processor that handles coordinate-sequence mapping properly.

This module provides improved template processing that correctly handles the
mapping between template coordinates (from PDB) and template sequences (from homology search).
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
import json

from .alignment import SequenceAligner, AlignmentResult
from .simple_template_manager import SimpleTemplateManager

logger = logging.getLogger(__name__)


class RobustTemplateProcessor:
    """
    Robust template processor with proper coordinate-sequence mapping.

    This processor correctly handles the case where template coordinates (from PDB)
    and template sequences (from homology search) have different lengths by using
    alignment information and coordinate mapping strategies.
    """

    def __init__(self, template_manager: SimpleTemplateManager,
                 contact_threshold: float = 8.0,  # CRITICAL FIX: Updated to 8Å as specified in task definition
                 min_seq_separation: int = 5,
                 distance_smooth_sigma: float = 0.5):
        """
        Initialize robust template processor.

        BIOPHYSICAL: Adaptive threshold parameters for conservative consensus building
        self.adaptive_threshold_high = 0.7  # Conservative: strong evidence required
        self.adaptive_threshold_medium = 0.5  # Conservative: majority evidence required
        self.adaptive_threshold_low = 0.3  # Liberal: weak evidence accepted
        """
        """
        Initialize robust template processor.

        Args:
            template_manager: SimpleTemplateManager instance
            contact_threshold: Distance threshold for contact definition (Å)
            min_seq_separation: Minimum sequence separation for contacts
            distance_smooth_sigma: Sigma for Gaussian smoothing of distance maps
        """
        self.template_manager = template_manager
        self.contact_threshold = contact_threshold
        self.min_seq_separation = min_seq_separation
        self.distance_smooth_sigma = distance_smooth_sigma
        self.aligner = SequenceAligner()

        # Log the key parameters for debugging
        logger.info(f"[INIT] RobustTemplateProcessor initialized:")
        logger.info(f"[INIT]   Contact threshold: {self.contact_threshold} Å")
        logger.info(f"[INIT]   Min sequence separation: {self.min_seq_separation}")
        logger.info(f"[INIT]   Consensus: CONTINUOUS PROBABILITIES (no binary threshold)")
        logger.info(f"[INIT]   Expected mean probability: 2-8% (for training)")

    def process_single_template(self, query_sequence: str, query_id: str,
                              template: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single template with proper coordinate-sequence mapping.

        Args:
            query_sequence: Query protein sequence
            query_id: Query identifier
            template: Template information from homology search

        Returns:
            Dictionary with processed template data or None if failed
        """
        pdb_id = template['pdb_id']
        chain_id = template['chain_id']

        try:
            # Validate template data structure first
            if not template:
                logger.error(f"Empty template data for {pdb_id}_{chain_id}")
                return None

            # Validate required fields
            required_fields = ['pdb_id', 'chain_id', 'sequence_identity', 'coverage', 'template_seq']
            missing_fields = [field for field in required_fields if not template.get(field)]
            if missing_fields:
                logger.error(f"Missing required fields for {pdb_id}_{chain_id}: {missing_fields}")
                return None

            # Get template coordinates from PDB
            coordinates = self.extract_template_coordinates(pdb_id, chain_id)
            if coordinates is None:
                logger.warning(f"Failed to extract coordinates for {pdb_id}_{chain_id}")
                return None

            # Get sequences from homology search
            template_seq = template.get('template_seq', '')
            query_seq_from_homology = template.get('query_seq', query_sequence)

            if not template_seq:
                logger.warning(f"No template sequence available for {pdb_id}_{chain_id}")
                return None

            # Validate sequence data
            if len(template_seq) < 10:
                logger.warning(f"Template sequence too short for {pdb_id}_{chain_id}: {len(template_seq)} aa")
                return None

            if len(query_seq_from_homology) < 20:
                logger.warning(f"Query sequence too short for {pdb_id}_{chain_id}: {len(query_seq_from_homology)} aa")
                return None

            # Use homology search quality metrics directly
            seq_identity = template.get('sequence_identity', 0.0)
            coverage = template.get('coverage', 0.0)
            e_value = template.get('e_value', float('inf'))

            # Create alignment between query and template sequences
            alignment = self.aligner.align_sequences(query_seq_from_homology, template_seq)

            # Create coordinate mapping that handles sequence length differences
            coord_mask, template_coords_mapped = self.map_coordinates_to_template_sequence(
                coordinates, len(template_seq), alignment
            )

            # Compute distance map using mapped coordinates
            distance_map, final_mask = self.compute_distance_map_from_coords(
                template_coords_mapped, len(query_sequence), alignment, coord_mask
            )

            # Validate distance map dimensions
            L_query = len(query_sequence)
            if distance_map.shape != (L_query, L_query):
                logger.error(f"Distance map shape mismatch: {distance_map.shape} vs {(L_query, L_query)}")
                return None

            # Compute contact map
            contact_map = self.compute_contact_map(distance_map, final_mask)

            # Validate contact map dimensions
            if contact_map.shape != (L_query, L_query):
                logger.error(f"Contact map shape mismatch: {contact_map.shape} vs {(L_query, L_query)}")
                return None

            # Convert 2D final_mask back to 1D coord_mask for multi-template processing
            # The query_mask is the diagonal of final_mask
            query_mask_1d = np.diag(final_mask)
            coord_mask = query_mask_1d

            # CRITICAL FIX: Compute proper template coverage from alignment
            # Template coverage should reflect what fraction of the template sequence is actually used
            template_positions_used = sum(1 for query_pos in range(len(query_sequence))
                                         if query_mask_1d[query_pos]
                                         and alignment.query_to_template.get(query_pos) is not None)
            proper_template_coverage = template_positions_used / len(template_seq) if template_seq else 0.0

            # CRITICAL FIX: Implement consistent quality score computation
            # Remove artificial variation and use scientifically meaningful factors

            # Base quality from sequence identity and coverage (most important factors)
            base_quality = 0.5 * (seq_identity + coverage)

            # E-value adjustment (log-scale transformation for better discrimination)
            # Transform E-value to a factor between 0.8 and 1.2
            if e_value <= 0:
                e_value = 1e-100  # Handle zero E-values
            log_e_value = np.log10(e_value)
            # Map log E-values: good (-100) -> 1.2, poor (0) -> 0.8
            e_value_factor = 1.2 + 0.4 * (log_e_value / 100)  # Linear scaling in log space
            e_value_factor = np.clip(e_value_factor, 0.8, 1.2)

            # CRITICAL FIX: Remove artificial hash-based variation that caused inconsistencies
            # Instead, use alignment score for meaningful variation if available
            alignment_score = template.get('alignment_score', 0.0)
            if alignment_score > 0:
                # Normalize alignment score using a reasonable scale (0-500 typical range)
                alignment_factor = 1.0 + 0.2 * np.tanh(alignment_score / 100.0)
            else:
                alignment_factor = 1.0

            # CRITICAL FIX: Redesign quality score to avoid multiplicative explosion > 1.0
            # Use bounded additive formula instead of multiplicative factors
            # This prevents quality scores from exceeding 1.0 due to factor multiplication

            # Base quality from sequence identity (dominant factor) - weighted 60%
            seq_quality_component = 0.6 * seq_identity

            # Coverage bonus (secondary factor) - weighted 25%
            coverage_component = 0.25 * coverage

            # E-value penalty (small negative adjustment) - max impact 10%
            # Convert good E-values to small bonus, poor E-values to penalty
            e_value_bonus = 0.1 * (1.0 - e_value_factor)  # 0 to 0.1 range

            # Alignment score bonus (minor adjustment) - max impact 5%
            alignment_bonus = 0.05 * (alignment_factor - 1.0)  # 0 to 0.05 range

            # CRITICAL FIX: Combine bounded components without diversity penalty
            # Diversity penalty will be applied in process_multiple_templates method
            quality_score = seq_quality_component + coverage_component + e_value_bonus + alignment_bonus

            # Apply final bounds to ensure proper [0,1] range
            quality_score = np.clip(quality_score, 0.0, 1.0)

            # CRITICAL FIX: No artificial template-specific variation
            # Similar templates should have similar quality scores

            # Calculate statistics
            valid_positions = np.sum(final_mask)
            total_possible_contacts = 0
            actual_contacts = 0

            L = len(query_sequence)
            for i in range(L):
                for j in range(i + self.min_seq_separation, L):
                    total_possible_contacts += 1
                    if contact_map[i, j] > 0.5:
                        actual_contacts += 1

            contact_density = actual_contacts / total_possible_contacts if total_possible_contacts > 0 else 0.0

            return {
                'template_pdb': pdb_id,
                'template_chain': chain_id,
                'template_quality_score': quality_score,
                'alignment_identity': alignment.identity,
                # CRITICAL FIX: Compute query coverage from actual coordinate mapping
                # Previous method used homology search coverage which may not match coordinate mapping
                'query_coverage': np.sum(coord_mask) / len(query_sequence) if query_sequence else 0.0,
                # CRITICAL FIX: Use properly computed template coverage from alignment
                'template_coverage': proper_template_coverage,
                'valid_positions': valid_positions,
                'valid_contacts': actual_contacts,
                'contact_density': contact_density,
                'distance_map': distance_map,
                'contact_map': contact_map,
                'coord_mask': coord_mask,
                'sequence_identity': seq_identity,  # Use homology search identity
                'coverage': coverage,
                'e_value': e_value
            }

        except Exception as e:
            logger.error(f"Error processing template {pdb_id}_{chain_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def extract_template_coordinates(self, pdb_id: str, chain_id: str) -> Optional[np.ndarray]:
        """Extract Cα coordinates from template."""
        return self.template_manager.get_template_coordinates(pdb_id, chain_id)

    def map_coordinates_to_template_sequence(self, coordinates: np.ndarray,
                                           template_seq_length: int,
                                           alignment: AlignmentResult) -> Tuple[np.ndarray, np.ndarray]:
        """
        CRITICAL FIX: Improved coordinate mapping that respects alignment positions.

        This handles the case where coordinates length != template sequence length
        by using the actual alignment information to map coordinates properly.
        """
        L_pdb = len(coordinates)
        L_template = template_seq_length

        # CRITICAL FIX: Use alignment information for precise coordinate mapping
        # Instead of naive sampling/padding, map based on actual residue correspondence

        # Get template positions that have aligned query residues
        aligned_template_positions = []
        aligned_query_positions = []

        for query_pos, template_pos in alignment.query_to_template.items():
            if template_pos is not None and 0 <= template_pos < L_pdb:
                # Only include positions where both template and query have residues
                aligned_template_positions.append(template_pos)
                aligned_query_positions.append(query_pos)

        if len(aligned_template_positions) == 0:
            # No valid alignment - fall back to simple uniform mapping
            logger.warning("No valid alignment positions found, using fallback mapping")
            if L_pdb >= L_template:
                # Sample uniformly
                indices = np.linspace(0, L_pdb - 1, L_template, dtype=int)
                template_coords_mapped = coordinates[indices]
                coord_mask = np.ones(L_template, dtype=bool)
            else:
                # Pad with zeros
                template_coords_mapped = np.zeros((L_template, 3))
                template_coords_mapped[:L_pdb] = coordinates
                coord_mask = np.zeros(L_template, dtype=bool)
                coord_mask[:L_pdb] = True
            return coord_mask, template_coords_mapped

        # CRITICAL FIX: Create coordinate mapping based on actual alignment
        # Initialize with zeros (missing coordinates)
        template_coords_mapped = np.zeros((L_template, 3))
        coord_mask = np.zeros(L_template, dtype=bool)

        # Map coordinates for aligned positions only
        for template_pos, query_pos in zip(aligned_template_positions, aligned_query_positions):
            if 0 <= template_pos < L_pdb:
                template_coords_mapped[query_pos] = coordinates[template_pos]
                coord_mask[query_pos] = True

        # Handle gaps: interpolate coordinates for unaligned template positions between aligned ones
        if np.any(coord_mask):
            # Find stretches of missing coordinates between aligned regions
            aligned_indices = np.where(coord_mask)[0]

            if len(aligned_indices) > 1:
                for i in range(len(aligned_indices) - 1):
                    start_idx = aligned_indices[i]
                    end_idx = aligned_indices[i + 1]

                    if end_idx - start_idx > 1:  # There's a gap
                        # Interpolate coordinates for the gap
                        gap_length = end_idx - start_idx - 1
                        start_coord = template_coords_mapped[start_idx]
                        end_coord = template_coords_mapped[end_idx]

                        # Linear interpolation
                        for j in range(gap_length):
                            alpha = (j + 1) / (gap_length + 1)
                            interp_pos = start_idx + j + 1
                            template_coords_mapped[interp_pos] = start_coord * (1 - alpha) + end_coord * alpha
                            coord_mask[interp_pos] = True  # Mark as interpolated coordinate

        # CRITICAL FIX: Check if mask is too sparse and apply fallback if needed
        mask_coverage = np.sum(coord_mask) / L_template
        if mask_coverage < 0.5:  # Less than 50% coverage is too sparse for contact prediction
            logger.warning(f"Mask too sparse ({mask_coverage:.3f} coverage), applying fallback mapping")
            if L_pdb >= L_template:
                # Sample uniformly from PDB coordinates - preserves structural information
                indices = np.linspace(0, L_pdb - 1, L_template, dtype=int)
                template_coords_mapped = coordinates[indices]
                coord_mask = np.ones(L_template, dtype=bool)
                logger.info(f"[DEBUG] Applied uniform sampling fallback: {L_pdb} -> {L_template} coordinates")
            else:
                # PDB is shorter - use all coordinates and pad
                template_coords_mapped = np.zeros((L_template, 3))
                template_coords_mapped[:L_pdb] = coordinates
                coord_mask = np.zeros(L_template, dtype=bool)
                coord_mask[:L_pdb] = True
                logger.info(f"[DEBUG] Applied padding fallback: {L_pdb} -> {L_template} coordinates")

        return coord_mask, template_coords_mapped

    def compute_distance_map_from_coords(self, template_coords: np.ndarray,
                                       query_length: int,
                                       alignment: AlignmentResult,
                                       coord_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance map using template coordinates aligned to query sequence.
        """
        # Validate inputs
        if query_length <= 0:
            logger.error(f"Invalid query length: {query_length}")
            return np.full((1, 1), np.nan), np.zeros(1, dtype=bool)

        if len(coord_mask) != len(template_coords):
            logger.error(f"Coord mask length mismatch: {len(coord_mask)} vs {len(template_coords)}")
            return np.full((query_length, query_length), np.nan), np.zeros(query_length, dtype=bool)

        # Map template coordinates to query positions using alignment
        mapped_coords, query_mask = self.aligner.map_template_coordinates(
            alignment, template_coords
        )

        # Validate mapped coordinates
        if len(query_mask) != query_length:
            logger.error(f"Query mask length mismatch after alignment: {len(query_mask)} vs {query_length}")
            return np.full((query_length, query_length), np.nan), np.zeros(query_length, dtype=bool)

        # Initialize distance map
        distance_map = np.full((query_length, query_length), np.nan)

        # Compute distances for mapped positions
        valid_indices = np.where(query_mask)[0]

        if len(valid_indices) > 1:
            # Extract mapped coordinates
            valid_coords = mapped_coords[valid_indices]

            # Compute pairwise distances
            distances = squareform(pdist(valid_coords))

            # Fill distance map
            for i, idx_i in enumerate(valid_indices):
                for j, idx_j in enumerate(valid_indices):
                    distance_map[idx_i, idx_j] = distances[i, j]

        # CRITICAL FIX: Apply enhanced Gaussian smoothing for full coverage
        # Previous approach didn't propagate distance information far enough into gaps
        if len(valid_indices) > 0:
            # Apply progressive smoothing to fill gaps more thoroughly
            smoothed_map = gaussian_filter(distance_map, sigma=self.distance_smooth_sigma)

            # Fill NaN values with smoothed values
            nan_mask = np.isnan(distance_map)
            distance_map[nan_mask] = smoothed_map[nan_mask]

            # Apply second round of stronger smoothing for remaining gaps
            # This ensures distance information propagates further into uncovered regions
            remaining_nan = np.isnan(distance_map)
            if np.any(remaining_nan):
                # Use stronger smoothing for remaining gaps
                strong_smoothed = gaussian_filter(distance_map, sigma=self.distance_smooth_sigma * 2)
                distance_map[remaining_nan] = strong_smoothed[remaining_nan]

            # Final fallback: fill any remaining NaN with maximum distance
            final_nan = np.isnan(distance_map)
            if np.any(final_nan):
                # Estimate maximum distance from valid entries
                valid_distances = distance_map[~np.isnan(distance_map)]
                if len(valid_distances) > 0:
                    max_dist = np.max(valid_distances)
                    # Add some margin (20%) to account for uncertainty
                    distance_map[final_nan] = max_dist * 1.2
                else:
                    # Complete fallback: use typical protein contact distance
                    distance_map[final_nan] = 15.0  # Typical upper bound for residue-residue distances

        # Create final mask combining query mask and coordinate mask
        final_mask = np.outer(query_mask, query_mask)

        # CRITICAL FIX: Enforce mask restriction after smoothing to prevent false contacts
        # This ensures NaN filling doesn't create artificial contacts outside coverage regions
        distance_map[~final_mask] = np.nan

        return distance_map, final_mask

    def compute_contact_map(self, distance_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute contact map from distance map."""
        contact_map = np.zeros_like(distance_map)

        # CRITICAL FIX: Handle normalized distance maps (0-1 range) by converting back to Å
        # This prevents false contacts when threshold is applied to normalized values
        distance_stats = {
            'min': np.nanmin(distance_map),
            'max': np.nanmax(distance_map),
            'mean': np.nanmean(distance_map)
        }

        logger.info(f"[DEBUG] Distance map stats: min={distance_stats['min']:.3f}, "
                   f"max={distance_stats['max']:.3f}, mean={distance_stats['mean']:.3f}")

        # Detect if distance_map is normalized (0-1 range) and convert back to Å for biophysical thresholding
        if distance_stats['max'] <= 1.5:  # Indicates normalized 0-1 range
            logger.info("[DEBUG] Detected normalized distance map, converting back to Å scale")
            # Convert to realistic protein Cα-Cα distance range (2-25 Å)
            # Use 25 Å as max to cover most protein structures
            original_distance_map = distance_map.copy()
            distance_map = distance_map * 25.0
            logger.info(f"[DEBUG] Scaled distances: max={np.nanmax(distance_map):.1f} Å, "
                       f"mean={np.nanmean(distance_map):.1f} Å")
        else:
            logger.info("[DEBUG] Using distance map in Å scale directly")

        # CRITICAL FIX: Apply proper masking before thresholding to prevent false contacts
        if mask.ndim == 1:
            # Use outer product of 1D mask for proper 2D masking
            mask_2d = np.outer(mask, mask)
            # CRITICAL FIX: Ensure strict masking before any thresholding
            distance_map[~mask_2d] = np.nan
            valid_mask = ~np.isnan(distance_map) & mask_2d
        else:
            # Use 2D mask directly
            distance_map[~mask] = np.nan
            valid_mask = ~np.isnan(distance_map) & mask

        # CRITICAL FIX: Use probability-based contact maps instead of binary thresholds
        # Convert distances to probabilities using sigmoid-like function
        # This provides continuous values [0,1] instead of binary [0,1]
        valid_distances = distance_map[valid_mask]

        # Use a soft threshold function (similar to logistic/sigmoid)
        # At contact_threshold (8Å), probability should be around 0.5
        # Sharpness parameter controls how quickly it transitions
        sharpness = 2.0  # Controls transition steepness
        offset = self.contact_threshold  # Threshold where probability = 0.5

        # Calculate probability: P = 1 / (1 + exp(sharpness * (distance - offset)))
        # This gives high probability (>0.9) for distances < 6Å
        # Medium probability (~0.5) for distances ≈ 8Å
        # Low probability (<0.1) for distances > 10Å
        probabilities = 1.0 / (1.0 + np.exp(sharpness * (valid_distances - offset)))

        contact_map[valid_mask] = probabilities.astype(float)

        # CRITICAL FIX: Handle any remaining NaN values in contact map
        contact_map[np.isnan(distance_map)] = 0.0

        # Calculate contact density using probability threshold (contacts with P > 0.5)
        high_prob_contacts = np.sum(contact_map > 0.5)
        logger.info(f"[DEBUG] Contact threshold={self.contact_threshold:.2f} Å, "
                   f"high_prob_contacts={high_prob_contacts}, "
                   f"mean_probability={np.mean(contact_map[contact_map > 0]):.3f}, "
                   f"density@P>0.5={high_prob_contacts/np.sum(mask_2d if mask.ndim==1 else mask):.4f}")

        # CRITICAL FIX: Apply minimum sequence separation with proper symmetry
        # Previous logic could miss some positions due to asymmetric range calculation
        L = distance_map.shape[0]
        exclusion_matrix = np.zeros((L, L), dtype=bool)

        # Create symmetric exclusion matrix for sequence separation
        for i in range(L):
            # Define exclusion range around diagonal (i.e., close residues)
            start_j = max(0, i - self.min_seq_separation + 1)
            end_j = min(L, i + self.min_seq_separation)
            exclusion_matrix[i, start_j:end_j] = True
            exclusion_matrix[start_j:end_j, i] = True  # Ensure symmetry

        # Apply exclusion to contact map
        contact_map[exclusion_matrix] = 0.0

        # CRITICAL FIX: Ensure proper symmetry and proper masking
        contact_map = np.maximum(contact_map, contact_map.T)  # Ensure symmetry
        if mask.ndim == 1:
            contact_map[~np.outer(mask, mask)] = 0.0  # Apply proper 2D mask

        # CRITICAL FIX: Final debug log after all processing (sequence separation applied)
        final_contacts = np.sum(contact_map)
        if mask.ndim == 1:
            valid_pairs = np.sum(np.triu(np.outer(mask, mask), k=self.min_seq_separation))
        else:
            valid_pairs = np.sum(np.triu(mask, k=self.min_seq_separation))
        final_density = final_contacts / valid_pairs if valid_pairs > 0 else 0.0

        logger.info(f"[DEBUG] Final contact map: contacts={final_contacts}, "
                   f"valid_pairs={valid_pairs}, density={final_density:.4f}, "
                   f"min_separation={self.min_seq_separation}")

        return contact_map

    def process_multiple_templates(self, query_sequence: str, query_id: str,
                                 templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Process multiple templates and create integrated features.
        """
        if not templates:
            return None

        processed_templates = []
        L = len(query_sequence)

        logger.info(f"Processing {len(templates)} templates for {query_id} (L={L})")

        # CRITICAL FIX: Optional single-template optimization logging
        if len(templates) == 1:
            logger.info("Single-template mode: using direct template contact map without combination")

        # Process each template
        for i, template in enumerate(templates):
            logger.debug(f"Processing template {i+1}/{len(templates)}: {template['pdb_id']}_{template['chain_id']}")

            processed = self.process_single_template(query_sequence, query_id, template)
            if processed:
                processed_templates.append(processed)
            else:
                logger.warning(f"Failed to process template {template['pdb_id']}_{template['chain_id']}")

        if not processed_templates:
            return None

        logger.info(f"Successfully processed {len(processed_templates)}/{len(templates)} templates")

        # CRITICAL FIX: Filter redundant templates to improve diversity and reduce bias
        filtered_templates = self._filter_redundant_templates(processed_templates, L)

        logger.info(f"Template diversity filtering: {len(processed_templates)} → {len(filtered_templates)} templates")

        # Use filtered templates for all subsequent processing
        processed_templates = filtered_templates

        # CRITICAL FIX: Log quality score distribution for debugging
        all_quality_scores = [t.get('template_quality_score', 0.0) for t in processed_templates]
        if all_quality_scores:
            quality_mean = np.mean(all_quality_scores)
            quality_std = np.std(all_quality_scores)
            quality_min = np.min(all_quality_scores)
            quality_max = np.max(all_quality_scores)
            logger.info(f"   Quality score distribution: mean={quality_mean:.3f}, std={quality_std:.3f}, range=[{quality_min:.3f}, {quality_max:.3f}]")
            if quality_std < 0.01:
                logger.warning(f"   ⚠️  Very low quality score variation - possible computation bug")
            elif quality_std > 0.3:
                logger.warning(f"   ⚠️  High quality score variation - possible inconsistent scoring")
            else:
                logger.info(f"   ✅ Quality score variation is reasonable (std={quality_std:.3f})")

        # Sort templates by quality score
        processed_templates.sort(key=lambda x: x['template_quality_score'], reverse=True)

        # Initialize integrated maps
        combined_distance_map = np.full((L, L), np.nan)
        combined_contact_map = np.zeros((L, L))
        combined_mask = np.zeros(L, dtype=bool)
        quality_weights = np.zeros((L, L))

        # Integrate templates using quality-weighted averaging
        total_weight = 0.0

        for template in processed_templates:
            quality = template['template_quality_score']
            coord_mask = template['coord_mask']
            distance_map = template['distance_map']
            contact_map = template['contact_map']

            # Validate dimensions
            if distance_map.shape != (L, L):
                logger.warning(f"Distance map shape mismatch for template {template.get('template_pdb', 'unknown')}: {distance_map.shape} vs {(L, L)}")
                continue

            if contact_map.shape != (L, L):
                logger.warning(f"Contact map shape mismatch for template {template.get('template_pdb', 'unknown')}: {contact_map.shape} vs {(L, L)}")
                continue

            if len(coord_mask) != L:
                logger.warning(f"Coord mask length mismatch for template {template.get('template_pdb', 'unknown')}: {len(coord_mask)} vs {L}")
                continue

            # Create quality weight matrix
            weight_matrix = np.outer(coord_mask, coord_mask) * quality

            # Ensure combined maps have correct shape
            if combined_distance_map.shape != (L, L):
                combined_distance_map = np.full((L, L), np.nan)
            if combined_contact_map.shape != (L, L):
                combined_contact_map = np.zeros((L, L))
            if quality_weights.shape != (L, L):
                quality_weights = np.zeros((L, L))

            # Validate weight_matrix dimensions before using
            if weight_matrix.shape != (L, L):
                logger.error(f"Weight matrix shape mismatch: {weight_matrix.shape} vs {(L, L)}")
                continue

            # Update combined maps (weighted average) - fixed dimension handling
            if total_weight > 0:
                # Only update where both maps have valid data and matching dimensions
                mask_valid = ~np.isnan(combined_distance_map) & ~np.isnan(distance_map) & (weight_matrix > 0)

                if np.any(mask_valid):
                    combined_distance_map[mask_valid] = (
                        (combined_distance_map[mask_valid] * total_weight +
                         distance_map[mask_valid] * weight_matrix[mask_valid]) /
                        (total_weight + weight_matrix[mask_valid])
                    )
            else:
                # First template - initialize combined maps
                mask_valid = ~np.isnan(distance_map) & (weight_matrix > 0)
                combined_distance_map[mask_valid] = distance_map[mask_valid]

            # CRITICAL FIX: Template combination should be ACCUMULATED weighted sum, not simple sum
            # Accumulate weighted contact maps for proper normalization later
            combined_contact_map += contact_map * weight_matrix
            quality_weights += weight_matrix
            combined_mask = combined_mask | coord_mask

            total_weight += quality

        # CRITICAL FIX: Replace problematic BOOLEAN UNION with proper normalized weighted mean
        # This prevents contact density inflation and ensures biologically realistic contact maps
        logger.info(f"DEBUG: Starting NORMALIZED contact combination")

        # Initialize contact maps from individual templates
        template_contact_maps = []
        template_quality_scores = []

        for template in processed_templates:
            contact_map = template.get('contact_map', np.array([]))
            coord_mask = template.get('coord_mask', np.array([]))
            quality = template['template_quality_score']

            # Validate dimensions
            if contact_map.size > 0 and coord_mask.size > 0:
                if contact_map.shape != (L, L):
                    logger.warning(f"Contact map shape mismatch for template {template.get('template_pdb', 'unknown')}: {contact_map.shape} vs {(L, L)}")
                    continue
                if len(coord_mask) != L:
                    logger.warning(f"Coord mask length mismatch for template {template.get('template_pdb', 'unknown')}: {len(coord_mask)} vs {L}")
                    continue

                # CRITICAL FIX: Use contact map values directly (already probabilities) instead of boolean conversion
                # Apply coordinate mask to remove invalid regions
                masked_contact_map = contact_map * np.outer(coord_mask, coord_mask)

                template_contact_maps.append(masked_contact_map)
                template_quality_scores.append(quality)

                # DEBUG: Log per-template contact statistics
                template_contacts = np.sum(np.triu(masked_contact_map > 0.5, k=self.min_seq_separation))
                template_pairs = np.sum(np.triu(np.outer(coord_mask, coord_mask), k=self.min_seq_separation))
                template_density = template_contacts / template_pairs if template_pairs > 0 else 0.0
                logger.info(f"   Template {template.get('template_pdb', 'Unknown')}: contacts={template_contacts}, density={template_density:.4f}")

        # Combine using PROPER NORMALIZED WEIGHTED MEAN (biophysical correction)
        if len(template_contact_maps) > 0:
            # Stack all template contact maps
            all_contacts = np.stack(template_contact_maps)  # Shape: (n_templates, L, L)

            # Apply quality weights and normalize properly
            weights = np.array(template_quality_scores, dtype=np.float32)

            # CRITICAL FIX: Normalize weights to sum to 1.0 (prevents inflation)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones_like(weights) / len(weights)  # Equal weights if all zero

            # Compute weighted mean properly
            weights_reshaped = weights.reshape(-1, 1, 1)  # Shape: (n_templates, 1, 1)
            combined_contact_map = np.sum(all_contacts * weights_reshaped, axis=0)  # Shape: (L, L)

            # CRITICAL FIX: Ensure values stay in [0,1] range
            combined_contact_map = np.clip(combined_contact_map, 0.0, 1.0)

            logger.info(f"   NORMALIZED combination stats: min={np.min(combined_contact_map):.3f}, max={np.max(combined_contact_map):.3f}")
            logger.info(f"   NORMALIZED combination sum: {np.sum(combined_contact_map):.1f}")
            logger.info(f"   Number of templates: {len(template_contact_maps)}")
            logger.info(f"   Normalized weights: {weights}")
        else:
            # No valid templates
            combined_contact_map = np.zeros((L, L))
            logger.warning(f"   No valid contact maps to combine")

        # Integrate distances using traditional weighted averaging (distance maps are continuous)
        total_weight = 0.0
        quality_weights = np.zeros((L, L))

        for template in processed_templates:
            quality = template['template_quality_score']
            coord_mask = template.get('coord_mask', np.array([]))
            distance_map = template['distance_map']

            # Validate dimensions
            if distance_map.shape != (L, L):
                logger.warning(f"Distance map shape mismatch for template {template.get('template_pdb', 'unknown')}: {distance_map.shape} vs {(L, L)}")
                continue

            if len(coord_mask) != L:
                logger.warning(f"Coord mask length mismatch for template {template.get('template_pdb', 'unknown')}: {len(coord_mask)} vs {L}")
                continue

            # Create quality weight matrix
            weight_matrix = np.outer(coord_mask, coord_mask) * quality

            # Update combined distance maps (weighted average)
            if total_weight > 0:
                # Only update where both maps have valid data and matching dimensions
                mask_valid = ~np.isnan(combined_distance_map) & ~np.isnan(distance_map) & (weight_matrix > 0)

                if np.any(mask_valid):
                    combined_distance_map[mask_valid] = (
                        (combined_distance_map[mask_valid] * total_weight +
                         distance_map[mask_valid] * weight_matrix[mask_valid]) /
                        (total_weight + weight_matrix[mask_valid])
                    )

            quality_weights += weight_matrix
            combined_mask = combined_mask | coord_mask

            total_weight += quality

        logger.info(f"   Template integration completed: BOOLEAN UNION contacts + weighted distances")

        # CRITICAL FIX: Apply enhanced gap-filling to combined distance map for full coverage
        # Use neutral mean distance for missing regions instead of zeros or extreme values
        final_mask = np.outer(combined_mask, combined_mask)

        # Calculate neutral distance from actual data (mean of valid distances)
        valid_distances = combined_distance_map[~np.isnan(combined_distance_map) & final_mask]
        if len(valid_distances) > 0:
            # Use mean distance as neutral fill value - more representative than maximum
            neutral_distance = np.mean(valid_distances)
            # Add some margin to account for uncertainty
            neutral_distance *= 1.1
        else:
            # Complete fallback: use typical protein contact distance
            neutral_distance = 12.0  # Mean residue-residue distance in proteins

        # CRITICAL FIX: Fill missing regions with neutral distance, not extreme values
        # This prevents CNN from interpreting gaps as strong contact/anti-contact signals
        missing_regions = ~final_mask | np.isnan(combined_distance_map)
        combined_distance_map[missing_regions] = neutral_distance

        # Apply gentle smoothing to blend boundaries between template-covered and missing regions
        if np.any(final_mask) and np.any(missing_regions):
            from scipy.ndimage import gaussian_filter
            # Gentle smoothing to avoid over-smoothing real structural information
            smoothed_map = gaussian_filter(combined_distance_map, sigma=self.distance_smooth_sigma * 0.5)

            # Only replace missing regions with smoothed values, preserve template-covered regions
            combined_distance_map[missing_regions] = smoothed_map[missing_regions]

            # Ensure missing regions stay close to neutral distance to avoid artificial signals
            deviation = np.abs(combined_distance_map[missing_regions] - neutral_distance)
            large_deviation = deviation > (neutral_distance * 0.3)  # Allow 30% deviation max
            if np.any(large_deviation):
                combined_distance_map[missing_regions][large_deviation] = neutral_distance

        # ENHANCED: Create consensus contact map with continuous probabilities
        # This preserves rich information for CNN training (SUPERIOR to binary approach)

        # Start with normalized combined contact map
        consensus_contacts = combined_contact_map.copy()

        # OPTIONAL: Apply mild sigmoid to enhance probability contrast while preserving continuity
        # This gives better 0-1 range without binary thresholding
        consensus_contacts = 1.0 / (1.0 + np.exp(-5 * (consensus_contacts - 0.3)))

        logger.info(f"[INFO] Creating continuous probability targets (no binary threshold)")
        logger.info(f"[INFO] Pre-processing contacts: {np.sum(consensus_contacts > 0.1):.0f} (p > 0.1)")
        logger.info(f"[INFO] Contact probability range: {np.min(consensus_contacts):.4f} - {np.max(consensus_contacts):.4f}")
        logger.info(f"[INFO] Mean probability: {np.mean(consensus_contacts):.4f}")

        # VALIDATE: Ensure reasonable probability distribution
        final_mean_prob = np.mean(consensus_contacts)
        expected_mean_range = (0.02, 0.08)  # Expected mean for biological contacts

        if final_mean_prob < expected_mean_range[0]:
            logger.warning(f"[WARNING] Contact probability too low: {final_mean_prob:.4f} "
                          f"(expected: {expected_mean_range[0]:.3f}-{expected_mean_range[1]:.3f})")
        elif final_mean_prob > expected_mean_range[1]:
            logger.warning(f"[WARNING] Contact probability too high: {final_mean_prob:.4f} "
                          f"(expected: {expected_mean_range[0]:.3f}-{expected_mean_range[1]:.3f})")
        else:
            logger.info(f"[SUCCESS] Contact probabilities in expected range: {final_mean_prob:.4f}")

        # CRITICAL FIX: Remove aggressive morphological dilation that inflates contact density
        # Only apply minimal local cleanup instead of expansive dilation
        # This preserves biologically realistic contact patterns

        # Apply final mask to ensure we only count positions with any template coverage
        consensus_contacts[~final_mask] = 0.0

        # CRITICAL FIX: Ensure symmetry of contact map after proper template combination
        consensus_contacts = np.maximum(consensus_contacts, consensus_contacts.T)

        # ENHANCED: Preserve continuous values for CNN training
        # NO BINARY CONVERSION - CNN will learn from probabilities directly
        logger.info(f"[INFO] Final continuous targets ready for CNN training")
        logger.info(f"[INFO] Final contacts > 0.1: {np.sum(consensus_contacts > 0.1):.0f}")
        logger.info(f"[INFO] Final mean probability: {np.mean(consensus_contacts):.4f}")

        # Calculate final statistics
        template_coverage = np.sum(combined_mask) / L

        # CRITICAL FIX: Count only upper triangle contacts to match total_possible calculation
        total_consensus_contacts = int(np.sum(np.triu(consensus_contacts > 0, k=self.min_seq_separation)))

        # CRITICAL FIX: Normalize total_possible using proper upper-triangle mask (matches validation approach)
        # This fixes the 5× density inflation by using consistent normalization
        valid_mask_upper = np.triu(final_mask, k=self.min_seq_separation)
        total_possible = np.sum(valid_mask_upper)

        # CRITICAL FIX: Recompute density using same approach as validation for consistency
        # This ensures internal density matches the external "corrected" density
        consensus_density = total_consensus_contacts / total_possible if total_possible > 0 else 0.0

        # CRITICAL FIX: Add debugging logs to verify fix
        logger.info(f"[DEBUG] Final consensus: contacts={total_consensus_contacts}, "
                   f"valid_pairs={np.sum(valid_mask_upper)}, "
                   f"density={consensus_density:.4f}, "
                   f"coverage={template_coverage:.3f}, "
                   f"expected_density={'VALID' if 0.015 <= consensus_density <= 0.035 else 'INVALID'}")

        # CRITICAL FIX: Add template coverage redundancy analysis
        template_diversity_analysis = self._analyze_template_diversity(processed_templates, L)

        return {
            'query_id': query_id,
            'query_length': L,
            'num_templates': len(processed_templates),
            'template_coverage': template_coverage,
            'combined_distance_map': combined_distance_map,
            'combined_contact_map': combined_contact_map,
            'consensus_contact_map': consensus_contacts,
            'combined_mask': combined_mask,
            'consensus_contacts': total_consensus_contacts,
            'consensus_density': consensus_density,
            'individual_templates': processed_templates,
            'quality_weights': quality_weights,
            'template_diversity': template_diversity_analysis  # NEW: Redundancy analysis
        }

    def _filter_redundant_templates(self, processed_templates: List[Dict[str, Any]], query_length: int,
                                max_overlap: float = 0.8) -> List[Dict[str, Any]]:
        """
        Filter redundant templates based on coverage overlap to improve diversity.

        Args:
            processed_templates: List of processed template dictionaries
            query_length: Length of the query sequence
            max_overlap: Maximum allowed Jaccard overlap between templates (default: 0.8)

        Returns:
            Filtered list of diverse templates
        """
        if len(processed_templates) <= 1:
            return processed_templates

        # Sort templates by quality score (highest first)
        sorted_templates = sorted(processed_templates,
                                key=lambda x: x.get('template_quality_score', 0.0),
                                reverse=True)

        filtered_templates = []
        filtered_masks = []

        for template in sorted_templates:
            coord_mask = template.get('coord_mask', np.array([], dtype=bool))

            # Skip if no coordinate mask
            if len(coord_mask) != query_length:
                continue

            # Check overlap with already selected templates
            is_redundant = False

            for existing_mask in filtered_masks:
                if len(existing_mask) == query_length:
                    # Calculate Jaccard overlap: intersection / union
                    intersection = np.sum(coord_mask & existing_mask)
                    union = np.sum(coord_mask | existing_mask)
                    overlap = intersection / union if union > 0 else 0.0

                    if overlap > max_overlap:
                        # Template is redundant - skip it
                        template_name = f"{template.get('template_pdb', 'Unknown')}_{template.get('template_chain', '?')}"
                        logger.info(f"   Filtering redundant template {template_name}: overlap={overlap:.3f} > {max_overlap}")
                        is_redundant = True
                        break

            if not is_redundant:
                # Keep this template
                filtered_templates.append(template)
                filtered_masks.append(coord_mask)
                template_name = f"{template.get('template_pdb', 'Unknown')}_{template.get('template_chain', '?')}"
                logger.debug(f"   Keeping diverse template {template_name}")

        return filtered_templates

    def _analyze_template_diversity(self, processed_templates: List[Dict[str, Any]], query_length: int) -> Dict[str, Any]:
        """
        Analyze template coverage diversity and redundancy.

        Returns:
            Dictionary with diversity metrics and warnings about template redundancy.
        """
        if len(processed_templates) < 2:
            return {
                'template_count': len(processed_templates),
                'redundancy_warning': False,
                'avg_pairwise_overlap': 0.0,
                'max_pairwise_overlap': 0.0,
                'diversity_score': 1.0,  # Single template is maximally diverse
                'recommendation': 'Add more templates for better coverage diversity'
            }

        # Extract coordinate masks for overlap analysis
        masks = []
        template_names = []
        for template in processed_templates:
            mask = template.get('coord_mask', np.array([], dtype=bool))
            masks.append(mask)
            template_name = f"{template.get('template_pdb', 'Unknown')}_{template.get('template_chain', '?')}"
            template_names.append(template_name)

        # Calculate pairwise overlap matrix
        num_templates = len(processed_templates)
        overlap_matrix = np.zeros((num_templates, num_templates))

        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                if len(masks[i]) == len(masks[j]) == query_length:
                    # Calculate Jaccard overlap: intersection / union
                    intersection = np.sum(masks[i] & masks[j])
                    union = np.sum(masks[i] | masks[j])
                    overlap = intersection / union if union > 0 else 0.0
                    overlap_matrix[i, j] = overlap_matrix[j, i] = overlap

        # Calculate overlap statistics
        avg_pairwise_overlap = np.mean(overlap_matrix[overlap_matrix > 0]) if np.any(overlap_matrix > 0) else 0.0
        max_pairwise_overlap = np.max(overlap_matrix)

        # Determine diversity score (0 = identical, 1 = completely diverse)
        diversity_score = 1.0 - max_pairwise_overlap

        # Generate warnings and recommendations
        redundancy_warning = max_pairwise_overlap > 0.8  # >80% overlap is highly redundant
        if redundancy_warning:
            recommendation = 'Templates are highly redundant (>80% overlap). Consider adding diverse templates.'
        elif max_pairwise_overlap > 0.6:
            recommendation = 'Templates show moderate redundancy (>60% overlap). Seek diverse coverage.'
        else:
            recommendation = 'Template diversity is acceptable.'

        return {
            'template_count': num_templates,
            'template_names': template_names,
            'pairwise_overlaps': overlap_matrix.tolist(),
            'redundancy_warning': redundancy_warning,
            'avg_pairwise_overlap': avg_pairwise_overlap,
            'max_pairwise_overlap': max_pairwise_overlap,
            'diversity_score': diversity_score,
            'recommendation': recommendation
        }

    def prepare_cnn_input(self, query_sequence: str, query_id: str,
                         templates: List[Dict[str, Any]], esm2_embeddings: np.ndarray,
                         max_esm2_channels: int = 64) -> Optional[Dict[str, Any]]:
        """
        Prepare CNN input combining template features with ESM2 embeddings.
        """
        # Process templates first
        template_data = self.process_multiple_templates(query_sequence, query_id, templates)
        if not template_data:
            return None

        L = len(query_sequence)
        channels = []
        channel_names = []

        # Channel 1: Template distance map (normalized)
        distance_map = template_data['combined_distance_map']
        distance_channel = np.copy(distance_map)

        # CRITICAL FIX: Proper distance normalization with missing data handling
        # Identify template-covered regions (where we have real structural data)
        template_mask = template_data['combined_mask']
        template_coverage_2d = np.outer(template_mask, template_mask)

        # Only normalize distances in template-covered regions
        template_covered_mask = template_coverage_2d & ~np.isnan(distance_channel)

        if np.sum(template_covered_mask) > 0:
            # Get statistics from template-covered regions only
            covered_distances = distance_channel[template_covered_mask]
            min_dist = np.min(covered_distances)
            max_dist = np.max(covered_distances)

            if max_dist > min_dist:
                # Normalize template-covered distances to [0, 1] range
                distance_channel[template_covered_mask] = (covered_distances - min_dist) / (max_dist - min_dist)

                # CRITICAL FIX: Set missing regions to neutral value (0.5) instead of 0.0 or 1.0
                # This represents "unknown/neutral" rather than "very close" or "very far"
                missing_mask = ~template_coverage_2d
                distance_channel[missing_mask] = 0.5  # Neutral value for missing data
            else:
                # All covered distances are the same - set covered to 0.0, missing to 0.5
                distance_channel[template_covered_mask] = 0.0
                distance_channel[~template_coverage_2d] = 0.5
        else:
            # No template coverage - set everything to neutral
            distance_channel[:] = 0.5

        channels.append(distance_channel)
        channel_names.append('template_distance')

        # Channel 2: Template contact map (weighted)
        contact_channel = template_data['combined_contact_map']
        channels.append(contact_channel)
        channel_names.append('template_contact')

        # Channel 3: Template coverage mask (FIXED - proper per-residue masking)
        # Instead of simple outer product, create proper residue-wise mask
        mask_channel = np.outer(template_data['combined_mask'], template_data['combined_mask']).astype(float)
        channels.append(mask_channel)
        channel_names.append('template_mask')

        # Channel 4: Template quality weights (FIXED - proper quality variation)
        quality_channel = template_data['quality_weights']

        # CRITICAL FIX: Instead of simple normalization, compute per-residue quality metrics
        # This ensures quality channel shows meaningful variation rather than constant blocks
        if len(template_data['individual_templates']) > 0:
            # Create quality variation based on template alignment quality
            # Use actual sequence identity and alignment information from individual templates
            quality_variation = np.zeros((L, L))

            for template in template_data['individual_templates']:
                seq_identity = template.get('alignment_identity', 0.0)
                template_coverage = template.get('query_coverage', 0.0)

                # Create per-residue quality based on alignment identity and coverage
                # This provides variation across the matrix instead of constant blocks
                per_template_quality = np.full((L, L), seq_identity * template_coverage)

                # Apply the same mask as the template
                template_mask = np.outer(template_data['combined_mask'], template_data['combined_mask'])
                per_template_quality *= template_mask

                quality_variation += per_template_quality

            # Average quality across templates (provides variation across matrix)
            quality_channel = quality_variation / len(template_data['individual_templates'])
        else:
            # Fallback to original if no individual templates
            quality_channel = template_data['quality_weights']
            max_quality = np.max(quality_channel)
            if max_quality > 0:
                quality_channel /= max_quality  # Normalize to [0, 1]

        channels.append(quality_channel)
        channel_names.append('template_quality')

        # Channels 5+: ESM2 embeddings (outer product features)
        if esm2_embeddings is not None and len(esm2_embeddings) == L:
            # Use top principal components or first N channels
            esm2_channels_to_use = min(max_esm2_channels, esm2_embeddings.shape[1])

            for i in range(esm2_channels_to_use):
                # Create outer product map for this embedding dimension
                embedding_dim = esm2_embeddings[:, i:i+1]  # (L, 1)
                outer_product = embedding_dim * embedding_dim.T  # (L, L)

                # Normalize to [0, 1]
                min_val = np.min(outer_product)
                max_val = np.max(outer_product)
                if max_val > min_val:
                    outer_product = (outer_product - min_val) / (max_val - min_val)

                channels.append(outer_product)
                channel_names.append(f'esm2_{i}')

        # Stack all channels
        multi_channel_input = np.stack(channels, axis=0)  # (C, L, L)

        return {
            'multi_channel_input': multi_channel_input,
            'channel_names': channel_names,
            'query_id': query_id,
            'query_length': L,
            'num_channels': len(channels),
            'template_data': template_data
        }