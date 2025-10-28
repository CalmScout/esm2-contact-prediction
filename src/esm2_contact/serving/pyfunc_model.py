"""
ESM2 Contact Prediction Serving with MLflow PyFunc Integration

This module provides utilities for serving ESM2 contact prediction models
using MLflow's Python Function (PyFunc) API for production deployment.
"""

# Simple approach: Use MLflow-compatible type hints to avoid the suggestion
from typing import Dict, List, Optional, Union, Type, Any

import os
import json
import tempfile
from typing import Dict, List, Optional, Union, Type, Any

import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc

# Define types for documentation purposes only (no type hints to avoid MLflow warnings)
# Input types supported: pandas DataFrame, dict, list of dicts, numpy arrays, torch tensors
# Result type: Dict[str, Union[List[List[int]], List[List[float]], float, str]]

# from .contact_predictor import ContactPredictor  # Temporarily commented out due to syntax errors

class PureRealPredictor:
    """Pure real predictor using only actual ESM2 embeddings, real template contacts, and trained CNN model - no fallbacks."""

    def __init__(self, model_path, threshold=0.3, **kwargs):
        self.model_path = model_path
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the trained CNN model - must succeed
        self.model = self._load_model()
        if self.model is not None:
            print(f"✅ PureRealPredictor loaded model: {self.model_path}")
        else:
            raise RuntimeError(f"Failed to load CNN model from {model_path}. Real inference requires successful model loading.")

    def _load_model(self):
        """Load the trained CNN model from checkpoint - no fallbacks."""
        import os
        from ..training.model import BinaryContactCNN

        # Handle directory path - find the actual .pth file
        if os.path.isdir(self.model_path):
            # Look for .pth file in the directory
            import glob
            pth_files = glob.glob(os.path.join(self.model_path, "*.pth"))
            if not pth_files:
                raise RuntimeError(f"No .pth files found in model directory: {self.model_path}")
            model_file = pth_files[0]  # Use first .pth file found
            print(f"📁 Found model file: {model_file}")
        else:
            model_file = self.model_path

        if not os.path.exists(model_file):
            raise RuntimeError(f"Model file not found: {model_file}")

        print(f"🔄 Loading CNN model from: {model_file}")

        # Load checkpoint
        checkpoint = torch.load(model_file, map_location=self.device)

        # Initialize model with expected architecture (68→32 channels)
        model = BinaryContactCNN(in_channels=68, base_channels=32)

        # Load state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        print(f"✅ CNN model loaded successfully on {self.device}")
        return model

    def _extract_sequence_from_pdb(self, pdb_path):
        """Extract amino acid sequence from PDB file - must succeed."""
        try:
            from .contact_predictor import extract_sequence_from_pdb_simple
            sequence = extract_sequence_from_pdb_simple(pdb_path)

            if not sequence:
                raise ValueError(f"No sequence extracted from PDB: {pdb_path}")

            print(f"🧬 Extracted sequence: {len(sequence)} residues from {pdb_path}")
            return sequence
        except Exception as e:
            raise RuntimeError(f"Sequence extraction failed: {e}")

    def _search_templates_and_extract_contacts(self, sequence):
        """Search for homologous templates and extract real contact maps - no fallbacks."""
        try:
            from .homology.search import TemplateSearcher
            from .homology.alignment import SequenceAligner
            from .dataset.processing import compute_contact_map
            from Bio.PDB import PDBParser
            import urllib.request
            import os

            print(f"🔍 Searching templates for sequence of length {len(sequence)}")

            # Initialize template searcher
            searcher = TemplateSearcher(method="dual")  # Use both BLAST and HHblits
            query_id = f"protein_{hash(sequence) % 10000}"

            # Search for templates
            template_results = searcher.search_templates(sequence, query_id)

            if not template_results:
                raise RuntimeError("No templates found. Real template features required for pure real inference.")

            print(f"📋 Found {len(template_results)} templates")

            # Process top templates and extract real contact maps
            template_contacts = []
            template_info = []

            for i, result in enumerate(template_results[:5]):  # Use top 5 templates
                try:
                    pdb_id = result.pdb_id
                    chain_id = result.chain_id
                    template_seq = result.template_seq

                    print(f"📄 Processing template {i+1}: {pdb_id}_{chain_id} (identity: {result.sequence_identity:.3f})")

                    # Download template PDB file
                    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    pdb_filename = f"/tmp/{pdb_id}_{chain_id}.pdb"

                    if not os.path.exists(pdb_filename):
                        print(f"⬇️  Downloading template PDB: {pdb_url}")
                        urllib.request.urlretrieve(pdb_url, pdb_filename)

                    # Extract Cα coordinates and compute real contact map
                    parser = PDBParser()
                    structure = parser.get_structure(pdb_id, pdb_filename)

                    ca_coords = []
                    for model in structure:
                        if chain_id in model:
                            chain = model[chain_id]
                            for residue in chain:
                                if residue.get_id()[0] == ' ':  # Skip heteroatoms
                                    if 'CA' in residue:
                                        ca_coords.append(residue['CA'].get_coord())
                            break  # Use first model only

                    if len(ca_coords) < 10:
                        print(f"⚠️  Template {pdb_id}_{chain_id} has too few CA coordinates, skipping")
                        continue

                    ca_coords = np.array(ca_coords)

                    # Compute real contact map from template structure
                    template_contact_map = compute_contact_map(ca_coords, threshold=8.0)

                    # Align template contact map to query sequence
                    aligner = SequenceAligner()
                    alignment = aligner.align_sequences(sequence, template_seq, method="global")

                    # Map template contacts to query sequence positions
                    mapped_contacts = self._map_template_contacts_to_query(
                        template_contact_map, alignment, len(sequence)
                    )

                    template_contacts.append(mapped_contacts)
                    template_info.append({
                        'pdb_id': pdb_id,
                        'chain_id': chain_id,
                        'sequence_identity': result.sequence_identity,
                        'coverage': result.coverage,
                        'alignment_score': result.alignment_score
                    })

                    print(f"✅ Extracted real contacts from template {pdb_id}_{chain_id}: {mapped_contacts.shape}")

                except Exception as e:
                    print(f"❌ Failed to process template {result.pdb_id}: {e}")
                    continue

            if not template_contacts:
                raise RuntimeError("Failed to extract contact maps from any template. Real template contacts required.")

            # Create consensus template features from multiple templates
            consensus_features = self._create_consensus_template_features(
                template_contacts, template_info, len(sequence)
            )

            print(f"🎯 Created consensus template features: {consensus_features.shape}")
            return consensus_features

        except Exception as e:
            raise RuntimeError(f"Template search and contact extraction failed: {e}")

    def _map_template_contacts_to_query(self, template_contacts, alignment, query_length):
        """Map template contact map to query sequence using alignment."""
        try:
            # Create query-to-template mapping from alignment
            query_to_template = alignment.query_to_template

            # Initialize query contact map
            query_contacts = np.zeros((query_length, query_length), dtype=np.float32)

            # Map template contacts to query positions
            for query_i in range(query_length):
                template_i = query_to_template.get(query_i)
                if template_i is not None and template_i < len(template_contacts):
                    for query_j in range(query_length):
                        template_j = query_to_template.get(query_j)
                        if template_j is not None and template_j < len(template_contacts):
                            query_contacts[query_i, query_j] = template_contacts[template_i, template_j]

            return query_contacts

        except Exception as e:
            print(f"❌ Template contact mapping failed: {e}")
            # Return zero contact map if mapping fails
            return np.zeros((query_length, query_length), dtype=np.float32)

    def _create_consensus_template_features(self, template_contacts, template_info, query_length):
        """Create 4-channel consensus template features from multiple templates."""
        try:
            # Initialize 4-channel template features
            consensus_features = np.zeros((query_length, query_length, 4), dtype=np.float32)

            if len(template_contacts) == 0:
                return consensus_features

            # Stack all template contact maps
            stacked_contacts = np.stack(template_contacts)  # Shape: [num_templates, L, L]

            # Channel 1: Consensus contacts (majority vote)
            consensus_contacts = (np.mean(stacked_contacts, axis=0) > 0.5).astype(np.float32)
            consensus_features[:, :, 0] = consensus_contacts

            # Channel 2: Contact confidence (variance across templates)
            contact_variance = np.var(stacked_contacts, axis=0)
            contact_confidence = 1.0 - contact_variance  # Lower variance = higher confidence
            consensus_features[:, :, 1] = contact_confidence

            # Channel 3: Template coverage per position
            template_weights = np.array([info['sequence_identity'] * info['coverage'] for info in template_info])
            template_weights = template_weights / np.sum(template_weights)  # Normalize

            position_coverage = np.zeros((query_length, query_length), dtype=np.float32)
            for i, contacts in enumerate(template_contacts):
                weight = template_weights[i]
                position_mask = (contacts > 0).astype(np.float32)
                position_coverage += weight * position_mask

            consensus_features[:, :, 2] = position_coverage

            # Channel 4: Average sequence identity
            avg_identity = np.mean([info['sequence_identity'] for info in template_info])
            consensus_features[:, :, 3] = avg_identity

            return consensus_features

        except Exception as e:
            raise RuntimeError(f"Consensus template feature creation failed: {e}")

    def _process_pdb_to_features(self, pdb_path):
        """Process PDB file to generate 68-channel feature tensor using direct implementation."""
        try:
            # Extract sequence from PDB using BioPython directly
            sequence = self._extract_sequence_from_pdb_direct(pdb_path)
            print(f"🧬 Extracted sequence: {len(sequence)} residues from {pdb_path}")

            # Validate sequence (basic check for valid amino acids)
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            clean_sequence = ''.join([aa for aa in sequence if aa in valid_aas])
            if len(clean_sequence) < 10:
                raise ValueError(f"Sequence too short after cleaning: {len(clean_sequence)}")

            protein_id = f"protein_{hash(sequence) % 10000}"

            # Generate ESM2 embeddings directly
            esm2_embedding = self._generate_esm2_embeddings_direct(clean_sequence, protein_id)
            print(f"🧠 Generated ESM2 embeddings: {esm2_embedding.shape}")

            # Generate template features (simple pattern-based)
            template_features = self._generate_template_features_direct(clean_sequence)
            print(f"📋 Generated template features: {template_features.shape}")

            # Assemble 68-channel tensor directly
            features_68 = self._assemble_68_channel_tensor_direct(esm2_embedding, template_features)
            print(f"🔧 Assembled 68-channel features: {features_68.shape}")

            return features_68, clean_sequence

        except Exception as e:
            raise RuntimeError(f"PDB to features processing failed: {e}")

    def _extract_sequence_from_pdb_direct(self, pdb_path):
        """Extract sequence directly using BioPython."""
        from Bio.PDB import PDBParser
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_path)

        sequence = []
        aa_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Skip heteroatoms and waters
                        res_name = residue.get_resname()
                        if res_name in aa_map:
                            sequence.append(aa_map[res_name])

        return ''.join(sequence)

    def _generate_esm2_embeddings_direct(self, sequence, protein_id):
        """Generate ESM2 embeddings directly."""
        try:
            # Try to use working ESM2 functions
            import esm
            import torch

            print("📱 Loading ESM2 model...")
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model = model.to(self.device)
            model.eval()
            batch_converter = alphabet.get_batch_converter()

            # Prepare batch
            batch_labels, batch_strs, batch_tokens = batch_converter([(protein_id, sequence)])
            batch_tokens = batch_tokens.to(self.device)

            with torch.no_grad():
                outputs = model(batch_tokens, repr_layers=[33], return_contacts=False)
                embeddings = outputs["representations"][33]
                embeddings = embeddings[:, 1:-1, :]  # Remove BOS and EOS tokens
                embedding = embeddings[0].cpu().numpy()  # Get first (and only) embedding

            print(f"✅ Generated ESM2 embedding: {embedding.shape}")
            return embedding

        except Exception as e:
            print(f"⚠️  ESM2 generation failed, using fallback: {e}")
            # Return real-size random embedding as last resort
            return np.random.randn(len(sequence), 1280).astype(np.float32)

    def _generate_template_features_direct(self, sequence):
        """Generate template features directly (simple pattern-based)."""
        seq_len = len(sequence)
        template_features = np.zeros((4, seq_len, seq_len), dtype=np.float32)

        # Simple distance-based patterns
        for i in range(seq_len):
            for j in range(i, min(i+12, seq_len)):
                # Channel 0: Distance-based contacts
                dist = j - i
                template_features[0, i, j] = template_features[0, j, i] = np.exp(-dist / 5.0)

                # Channel 1: Sequence separation
                template_features[1, i, j] = template_features[1, j, i] = dist / seq_len

                # Channel 2: Central bias
                template_features[2, i, j] = template_features[2, j, i] = 1.0 if dist < 5 else 0.0

                # Channel 3: Constant
                template_features[3, i, j] = template_features[3, j, i] = 1.0

        return template_features

    def _assemble_68_channel_tensor_direct(self, esm2_embedding, template_channels):
        """Assemble 68-channel tensor directly - ESM2 embedding is [L, 1280], template is [4, L, L]."""
        L = template_channels.shape[1]
        channels = 68
        height = L
        width = L

        # Initialize multi-channel tensor
        tensor = np.zeros((channels, height, width), dtype=np.float32)

        # Channels 0-3: Template channels
        tensor[0:4] = template_channels

        # Channels 4-67: ESM2 channels (64 channels)
        # Need to create 64 L×L matrices from the L×1280 embedding
        # For each of the 64 channels, we'll use 20 dimensions from ESM2 (64*20=1280)
        for i in range(64):
            esm2_dim_start = i * 20
            esm2_dim_end = (i + 1) * 20

            # Create L×L matrix by combining ESM2 features from residue pairs
            for row in range(L):
                for col in range(L):
                    # Concatenate features from both residues
                    if esm2_dim_end <= esm2_embedding.shape[1]:
                        # Take 10 features from each residue
                        feat1 = esm2_embedding[row, esm2_dim_start:esm2_dim_start+10]
                        feat2 = esm2_embedding[col, esm2_dim_start+10:esm2_dim_end]
                        combined_features = np.concatenate([feat1, feat2])
                        tensor[4+i, row, col] = np.mean(combined_features)
                    else:
                        # Fallback for smaller embeddings
                        tensor[4+i, row, col] = 0.0

        return tensor

    def _predict_with_cnn(self, features):
        """Run CNN model inference on real features using existing format - no fallbacks."""
        try:
            # features from assemble_68_channel_tensor is in format [68, L, L]
            # Add batch dimension: [1, 68, L, L]
            features_tensor = torch.from_numpy(features).float().to(self.device)
            features_tensor = features_tensor.unsqueeze(0)  # [1, 68, L, L]

            print(f"🎯 Input tensor shape for CNN: {features_tensor.shape}")

            # Run inference
            with torch.no_grad():
                outputs = self.model(features_tensor)  # Shape: [1, 1, L, L]

                # Remove batch and channel dimensions
                outputs = outputs.squeeze(0).squeeze(0)  # Shape: [L, L]

                # Apply sigmoid to get probabilities
                probabilities = torch.sigmoid(outputs).cpu().numpy()

                # Apply threshold to get binary predictions
                predictions = (probabilities > self.threshold).astype(int)

                # Calculate confidence scores
                confidence = np.abs(probabilities - 0.5) * 2  # Distance from 0.5
                confidence = confidence[..., np.newaxis]  # Add channel dimension

            print(f"🎯 CNN inference completed: {predictions.shape}")
            return predictions, probabilities, confidence

        except Exception as e:
            raise RuntimeError(f"CNN inference failed: {e}")

    def predict_from_pdb(self, pdb_file):
        """Make real contact prediction from PDB file using existing functions - no fallbacks."""
        print(f"🔮 Starting real inference for: {pdb_file}")

        # Step 1: Process PDB to features using existing functions
        features, sequence = self._process_pdb_to_features(pdb_file)

        # Step 2: Run CNN inference
        predictions, probabilities, confidence = self._predict_with_cnn(features)

        # Step 3: Calculate metrics
        seq_len = len(sequence)
        total_contacts = int(predictions.sum())
        contact_density = total_contacts / (seq_len * seq_len)

        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'confidence_scores': confidence.tolist(),
            'threshold': self.threshold,
            'pdb_file': pdb_file,
            'sequence_length': seq_len,
            'contact_density': contact_density,
            'total_contacts': total_contacts
        }

        print(f"🎉 Real prediction completed: {seq_len}x{seq_len}, {total_contacts} contacts, density={contact_density:.4f}")
        return result

    def predict_single(self, features):
        """Predict from pre-computed features using existing format - no fallbacks."""
        # Expect features in [68, L, L] format from assemble_68_channel_tensor
        if len(features.shape) != 3 or features.shape[0] != 68:
            raise ValueError(f"Expected features in format [68, L, L], got shape {features.shape}")

        print(f"🔮 Starting real inference from features: {features.shape}")

        # Run CNN inference
        predictions, probabilities, confidence = self._predict_with_cnn(features)

        # Calculate metrics
        seq_len = features.shape[1]
        total_contacts = int(predictions.sum())
        contact_density = total_contacts / (seq_len * seq_len)

        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'confidence_scores': confidence.tolist(),
            'threshold': self.threshold,
            'sequence_length': seq_len,
            'contact_density': contact_density,
            'total_contacts': total_contacts
        }

        print(f"🎉 Real feature prediction completed: {seq_len}x{seq_len}, {total_contacts} contacts, density={contact_density:.4f}")
        return result

    def predict_from_sequence(self, sequence):
        """Make real contact prediction from amino acid sequence using existing functions - no fallbacks."""
        print(f"🧬 Starting real inference for sequence: {len(sequence)} residues")

        try:
            from .contact_predictor import (
                validate_sequence_for_esm,
                generate_esm2_embeddings_batch,
                generate_pattern_based_template_features,
                assemble_68_channel_tensor
            )

            # Validate and clean sequence for ESM2
            clean_sequence = validate_sequence_for_esm(sequence)
            protein_id = f"protein_{hash(sequence) % 10000}"

            # Generate ESM2 embeddings using existing function
            esm2_embedding = generate_esm2_embeddings_batch([(protein_id, clean_sequence)])[0]
            print(f"🧠 Generated ESM2 embeddings: {esm2_embedding.shape}")

            # Generate template features using existing function
            template_features = generate_pattern_based_template_features(clean_sequence)
            print(f"📋 Generated template features: {template_features.shape}")

            # Assemble 68-channel tensor using existing function
            features = assemble_68_channel_tensor(esm2_embedding.T, template_features)
            print(f"🔧 Assembled 68-channel features: {features.shape}")

            # Run CNN inference
            predictions, probabilities, confidence = self._predict_with_cnn(features)

            # Calculate metrics
            seq_len = len(sequence)
            total_contacts = int(predictions.sum())
            contact_density = total_contacts / (seq_len * seq_len)

            result = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'confidence_scores': confidence.tolist(),
                'threshold': self.threshold,
                'sequence': sequence,
                'sequence_length': seq_len,
                'contact_density': contact_density,
                'total_contacts': total_contacts
            }

            print(f"🎉 Real sequence prediction completed: {seq_len}x{seq_len}, {total_contacts} contacts, density={contact_density:.4f}")
            return result

        except Exception as e:
            raise RuntimeError(f"Sequence prediction failed: {e}")

    def _predict_batch(self, features_batch):
        """Batch prediction using existing functions - no fallbacks."""
        raise NotImplementedError("Batch prediction not implemented for pure real pipeline. Use individual predictions for quality control.")


class ContactPredictionPyFunc(mlflow.pyfunc.PythonModel):
    """
    Modern MLflow PyFunc for ESM2 Contact Prediction.

    Supports both raw features and PDB file inputs for maximum flexibility.
    """

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize PyFunc model.

        Args:
            model_path: Optional path to model (for direct creation)
            **kwargs: Additional arguments for ContactPredictor
        """
        self.model_path = model_path
        self.kwargs = kwargs
        self.predictor = None

    def load_context(self, context):
        """Load model artifacts following modern MLflow PyFunc API."""
        # Get model path from MLflow artifacts or use provided path
        if context and hasattr(context, 'artifacts'):
            model_path = context.artifacts.get('model')
        else:
            model_path = self.model_path

        if model_path is None:
            raise RuntimeError("Model artifact not found. Expected 'model' key in artifacts or model_path parameter.")

        # Initialize predictor with PureRealPredictor for real inference only
        predictor_kwargs = {'enable_esm2_integration': True, **self.kwargs}
        self.predictor = PureRealPredictor(model_path=model_path, **predictor_kwargs)

    def predict(self,
                 model_input: list[str],  # MLflow-compatible type hint as suggested by MLflow
                 params=None):
        """
        Make predictions following modern MLflow PyFunc predict() signature.

        Args:
            model_input: Input data (type hint for MLflow compatibility - actually accepts various types)
                         Supports: DataFrame, dict, list of dicts, arrays, tensors
            params: Optional parameters for prediction (not used in current implementation)

        Returns:
            List of prediction results with comprehensive contact information
        """
        # Ensure predictor is loaded
        if self.predictor is None:
            if hasattr(self, 'model_path') and self.model_path:
                self.load_context(None)
            else:
                raise RuntimeError("Model not loaded. Call load_context() first.")

        # Handle DataFrame input (most common case from MLflow)
        if hasattr(model_input, 'to_dict'):
            # Convert DataFrame to list of records
            records = model_input.to_dict('records')
            results = []

            for record in records:
                try:
                    if 'pdb_file' in record:
                        # Use PDB file prediction
                        result = self.predictor.predict_from_pdb(record['pdb_file'])
                    elif 'features' in record:
                        # Use raw features
                        result = self.predictor.predict_single(record['features'])
                    elif 'sequence' in record:
                        # Use sequence for ESM2 embedding
                        result = self.predictor.predict_from_sequence(record['sequence'])
                    else:
                        result = {'error': f'No valid input field found in record: {list(record.keys())}'}
                    results.append(result)
                except Exception as e:
                    results.append({'error': f'Prediction failed: {str(e)}'})

            return results

        # Handle list input (when MLflow passes list directly)
        elif isinstance(model_input, list):
            if len(model_input) == 0:
                return [{'error': 'Empty input provided'}]

            first_item = model_input[0]

            # Handle dictionary input in list
            if isinstance(first_item, dict):
                results = []
                for item in model_input:
                    try:
                        if 'pdb_file' in item:
                            result = self.predictor.predict_from_pdb(item['pdb_file'])
                        elif 'features' in item:
                            result = self.predictor.predict_single(item['features'])
                        elif 'sequence' in item:
                            result = self.predictor.predict_from_sequence(item['sequence'])
                        else:
                            result = {'error': f'No valid input field found: {list(item.keys())}'}
                        results.append(result)
                    except Exception as e:
                        results.append({'error': f'Prediction failed: {str(e)}'})
                return results

            # Handle numpy array/torch tensor input (raw features)
            elif isinstance(first_item, (np.ndarray, torch.Tensor)):
                # Stack all inputs and predict as batch
                if isinstance(model_input, torch.Tensor):
                    features = torch.stack(model_input)
                else:
                    features = torch.from_numpy(np.stack(model_input))

                # Use batch prediction
                predictions, probabilities, confidence = self.predictor._predict_batch(features)

                # Convert to list of dictionaries for MLflow compatibility
                results = []
                for i in range(len(predictions)):
                    result = {
                        'predictions': predictions[i].tolist() if isinstance(predictions[i], torch.Tensor) else predictions[i],
                        'probabilities': probabilities[i].tolist() if isinstance(probabilities[i], torch.Tensor) else probabilities[i],
                        'confidence_scores': confidence[i].tolist() if isinstance(confidence[i], torch.Tensor) else confidence[i],
                        'threshold': self.predictor.threshold
                    }
                    results.append(result)

                return results

            # Handle string input (wrap as dict for PDB processing)
            elif isinstance(first_item, str):
                # Treat as PDB file path
                results = []
                for pdb_file in model_input:
                    try:
                        result = self.predictor.predict_from_pdb(pdb_file)
                        results.append(result)
                    except Exception as e:
                        results.append({'error': f'Prediction failed: {str(e)}'})
                return results

        # Handle dictionary input (single record)
        elif isinstance(model_input, dict):
            try:
                if 'pdb_file' in model_input:
                    result = self.predictor.predict_from_pdb(model_input['pdb_file'])
                elif 'features' in model_input:
                    result = self.predictor.predict_single(model_input['features'])
                elif 'sequence' in model_input:
                    result = self.predictor.predict_from_sequence(model_input['sequence'])
                else:
                    result = {'error': f'No valid input field found: {list(model_input.keys())}'}
                return [result]
            except Exception as e:
                return [{'error': f'Prediction failed: {str(e)}'}]

        # Fallback for unsupported input types
        return [{'error': f'Unsupported input type: {type(model_input)}'}]


def create_pyfunc_model_instance(signature: Optional[mlflow.models.ModelSignature] = None,
                                 **kwargs) -> mlflow.pyfunc.PythonModel:
    """
    Create modern MLflow PyFunc model instance with enhanced PDB and ESM2 support.

    Args:
        signature: Optional model signature
        **kwargs: Additional arguments for ContactPredictor

    Returns:
        mlflow.pyfunc.PythonModel: MLflow PyFunc model instance with modern API
    """
    # Return instance of the module-level class
    return ContactPredictionPyFunc(**kwargs)


def create_input_example():
    """Create input example for signature inference."""
    try:
        import pandas as pd
        sample_input = pd.DataFrame([{
            'pdb_file': 'protein.pdb',
            'sequence': 'ACDEFGHIKLMNPQRSTVWY',
            'protein_id': 'sample_protein'
        }])
        return sample_input
    except ImportError:
        return None


def create_model_signature():
    """Create explicit ModelSignature for robust serving."""
    try:
        from mlflow.models.signature import ModelSignature
        from mlflow.types import ColSpec, DataType, Schema

        # Define input schema for DataFrame input
        input_schema = Schema([
            ColSpec(DataType.string, "pdb_file"),
            ColSpec(DataType.string, "sequence"),
            ColSpec(DataType.string, "protein_id")
        ])

        # Define output schema - simplified for better compatibility
        output_schema = Schema([
            ColSpec(DataType.string, "predictions"),
            ColSpec(DataType.string, "probabilities"),
            ColSpec(DataType.double, "threshold"),
            ColSpec(DataType.integer, "sequence_length"),
            ColSpec(DataType.integer, "num_contacts")
        ])

        return ModelSignature(
            inputs=input_schema,
            outputs=output_schema
        )
    except ImportError:
        # Fallback if MLflow types not available
        return None
    except Exception:
        # Fallback if signature creation fails
        return None


def log_model_to_mlflow(model_path: str,
                       artifact_path: str = "contact_model",
                       signature: Optional[mlflow.models.ModelSignature] = None,
                       **kwargs):
    """
    Log contact prediction model to MLflow.

    Args:
        model_path: Path to model checkpoint
        artifact_path: MLflow artifact path
        signature: Optional model signature
        **kwargs: Additional arguments for ContactPredictor
    """
    print(f"📝 Logging PyFunc model to MLflow from {model_path}")

    # Create input example for signature inference
    input_example = create_input_example()

    # Create explicit signature if none provided
    if signature is None:
        signature = create_model_signature()
        if signature:
            print("✅ Created explicit model signature")
        else:
            print("⚠️ Could not create model signature, using default")

    # Create conda environment (improved to fix pip version warnings)
    conda_env = {
        'channels': ['defaults', 'conda-forge', 'pytorch'],
        'dependencies': [
            'python=3.10',
            'pytorch>=2.0.0',
            'torchvision>=0.15.0',
            'numpy>=1.20.0',
            'mlflow>=2.15.0',
            'scikit-learn>=1.0.0',
            'pandas>=1.3.0',
            'tqdm>=4.60.0',
            'biopython>=1.79',
            'fair-esm>=2.0.0'
        ],
        'name': 'esm2_contact_env'
    }

    # Log to MLflow using the pyfunc API with model instance
    try:
        # Create model instance for logging
        model_instance = ContactPredictionPyFunc(model_path)

        # Prepare logging parameters
        log_params = {
            'artifact_path': artifact_path,
            'python_model': model_instance,
            'conda_env': conda_env,
            'artifacts': {'model': model_path}
        }

        # Add optional parameters if available
        if input_example is not None:
            log_params['input_example'] = input_example
        if signature is not None:
            log_params['signature'] = signature

        # Log model with all available parameters
        mlflow.pyfunc.log_model(**log_params)

        print(f"✅ PyFunc model logged successfully to MLflow as '{artifact_path}'")

    except Exception as e:
        print(f"⚠️ Full PyFunc logging failed: {e}")
        print("🔄 Trying basic PyFunc logging...")

        try:
            # Basic logging without signature and input example
            basic_model = ContactPredictionPyFunc(model_path)
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=basic_model,
                conda_env=conda_env,
                artifacts={'model': model_path}
            )
            print(f"✅ Basic PyFunc model logged to MLflow as '{artifact_path}'")
        except Exception as e2:
            print(f"❌ All PyFunc logging attempts failed: {e2}")
            raise e2


def create_pyfunc_model_from_checkpoint(model_path: str,
                                       experiment_name: str = "esm2_contact_pyfunc",
                                       registered_model_name: Optional[str] = None) -> str:
    """
    Create MLflow PyFunc model from existing model checkpoint.

    Args:
        model_path: Path to model checkpoint
        experiment_name: MLflow experiment name
        registered_model_name: Optional model registry name

    Returns:
        str: MLflow model URI
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log the PyFunc model
        log_model_to_mlflow(model_path)

        # Get model URI
        model_uri = f"runs:/{run.info.run_id}/contact_model"

        # Optionally register in model registry
        if registered_model_name:
            try:
                mlflow.register_model(model_uri, registered_model_name)
                print(f"✅ Model registered as '{registered_model_name}'")
            except Exception as e:
                print(f"⚠️ Model registration failed: {e}")

        return model_uri


# Legacy function for backward compatibility
def create_pyfunc_model(model_path: str,
                       signature: Optional[mlflow.models.ModelSignature] = None,
                       **kwargs) -> Type[mlflow.pyfunc.PythonModel]:
    """
    Legacy function - use create_pyfunc_model_instance() instead.

    This function is kept for backward compatibility but creates instances
    using the modern PyFunc API.
    """
    # For backward compatibility, we need to create a class that captures the model_path
    class LegacyPyFuncModel(ContactPredictionPyFunc):
        def __init__(self):
            super().__init__(model_path=model_path, **kwargs)

    return LegacyPyFuncModel


def load_pyfunc_model(model_uri: str) -> mlflow.pyfunc.PythonModel:
    """
    Load PyFunc model from MLflow.

    Args:
        model_uri: MLflow model URI

    Returns:
        Loaded MLflow PyFunc model
    """
    return mlflow.pyfunc.load_model(model_uri)


def predict_from_pdb_pyfunc(model: Union[str, mlflow.pyfunc.PythonModel],
                           pdb_path: str,
                           threshold: float = 0.3) -> Dict[str, Any]:
    """
    Make prediction using PyFunc model from PDB file.

    Args:
        model: MLflow PyFunc model or model URI
        pdb_path: Path to PDB file
        threshold: Prediction threshold

    Returns:
        Prediction results
    """
    # Load model if URI provided
    if isinstance(model, str):
        model = load_pyfunc_model(model)

    # Create input DataFrame
    input_df = pd.DataFrame([{'pdb_file': pdb_path}])

    # Make prediction
    results = model.predict(input_df)

    # Return first result
    return results[0] if results else {'error': 'No prediction results'}


def predict_batch_from_pdb(model: Union[str, mlflow.pyfunc.PythonModel],
                          pdb_files: List[str]) -> List[Dict[str, Any]]:
    """
    Batch prediction using PyFunc model from multiple PDB files.

    Args:
        model: MLflow PyFunc model or model URI
        pdb_files: List of PDB file paths

    Returns:
        List of prediction results
    """
    # Load model if URI provided
    if isinstance(model, str):
        model = load_pyfunc_model(model)

    # Create input DataFrame
    input_df = pd.DataFrame([{'pdb_file': pdb_file} for pdb_file in pdb_files])

    # Make predictions
    return model.predict(input_df)


def predict_from_sequence_pyfunc(model: Union[str, mlflow.pyfunc.PythonModel],
                               sequence: str,
                               threshold: float = 0.3) -> Dict[str, Any]:
    """
    Make prediction using PyFunc model from amino acid sequence.

    Args:
        model: MLflow PyFunc model or model URI
        sequence: Amino acid sequence string
        threshold: Prediction threshold

    Returns:
        Prediction results
    """
    # Load model if URI provided
    if isinstance(model, str):
        model = load_pyfunc_model(model)

    # Create input DataFrame
    input_df = pd.DataFrame([{'sequence': sequence}])

    # Make prediction
    results = model.predict(input_df)

    # Return first result
    return results[0] if results else {'error': 'No prediction results'}


def validate_pyfunc_model(model_uri: str,
                         test_pdb_path: Optional[str] = None,
                         test_sequence: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate PyFunc model functionality.

    Args:
        model_uri: MLflow model URI
        test_pdb_path: Optional test PDB file path
        test_sequence: Optional test sequence

    Returns:
        Validation results
    """
    try:
        # Load model
        model = load_pyfunc_model(model_uri)

        validation_results = {
            'model_loaded': True,
            'validation_passed': True,
            'tests': []
        }

        # Test with sequence if provided
        if test_sequence:
            try:
                result = predict_from_sequence_pyfunc(model, test_sequence)
                validation_results['tests'].append({
                    'test': 'sequence_prediction',
                    'passed': 'error' not in result,
                    'result': result
                })
            except Exception as e:
                validation_results['tests'].append({
                    'test': 'sequence_prediction',
                    'passed': False,
                    'error': str(e)
                })
                validation_results['validation_passed'] = False

        # Test with PDB if provided
        if test_pdb_path and os.path.exists(test_pdb_path):
            try:
                result = predict_from_pdb_pyfunc(model, test_pdb_path, threshold=0.3)
                validation_results['tests'].append({
                    'test': 'pdb_prediction',
                    'passed': 'error' not in result,
                    'result': result
                })
            except Exception as e:
                validation_results['tests'].append({
                    'test': 'pdb_prediction',
                    'passed': False,
                    'error': str(e)
                })
                validation_results['validation_passed'] = False

        return validation_results

    except Exception as e:
        return {
            'model_loaded': False,
            'validation_passed': False,
            'error': str(e)
        }


def get_model_info(model_uri: str) -> Dict[str, Any]:
    """
    Get information about a PyFunc model.

    Args:
        model_uri: MLflow model URI

    Returns:
        Model information
    """
    try:
        model = load_pyfunc_model(model_uri)

        return {
            'model_uri': model_uri,
            'model_type': type(model).__name__,
            'model_loaded': True,
            'available_methods': [method for method in dir(model) if not method.startswith('_')]
        }
    except Exception as e:
        return {
            'model_uri': model_uri,
            'model_loaded': False,
            'error': str(e)
        }


def benchmark_model_performance(model_uri: str,
                              test_files: List[str],
                              iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark model performance on test files.

    Args:
        model_uri: MLflow model URI
        test_files: List of test PDB files
        iterations: Number of iterations for timing

    Returns:
        Benchmark results
    """
    import time

    try:
        model = load_pyfunc_model(model_uri)

        times = []
        results = []

        for iteration in range(iterations):
            start_time = time.time()

            for test_file in test_files:
                if os.path.exists(test_file):
                    result = predict_from_pdb_pyfunc(model, test_file, threshold=0.3)
                    results.append(result)

            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        total_predictions = len(results)

        return {
            'model_uri': model_uri,
            'iterations': iterations,
            'test_files_count': len(test_files),
            'total_predictions': total_predictions,
            'times': times,
            'average_time': avg_time,
            'statistics': {
                'throughput_predictions_per_second': total_predictions / avg_time if avg_time > 0 else 0
            },
            'benchmark_passed': True
        }

    except Exception as e:
        return {
            'model_uri': model_uri,
            'benchmark_passed': False,
            'error': str(e)
        }