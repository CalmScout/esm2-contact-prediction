"""
Modern ESM2 Contact Prediction Serving Module

This module provides modern MLflow PyFunc integration for serving trained
protein contact prediction models in production environments.

Key Features:
- Modern MLflow PyFunc custom models with proper load_context() and predict() methods
- Production-ready inference pipeline with ESM2 integration
- Batch processing capabilities for multiple proteins
- PDB file processing capabilities
- Model versioning and registry support
- Enhanced error handling and monitoring

Components:
- ContactPredictor: Enhanced predictor with ESM2 integration
- create_pyfunc_model_instance(): Modern PyFunc model creation
- PDB processing utilities and batch inference

Usage:
    from esm2_contact.serving import ContactPredictor, create_pyfunc_model_instance

    # Create enhanced predictor
    model = ContactPredictor(model_path="model.pth", enable_esm2_integration=True)

    # Predict from PDB file
    results = model.predict_from_pdb("protein.pdb")

    # Create modern MLflow PyFunc for serving
    pyfunc_model = create_pyfunc_model_instance(model_path="model.pth")
    mlflow.pyfunc.log_model(pyfunc_model, "contact_predictor")

    # Load and serve with multiple input formats
    served_model = mlflow.pyfunc.load_model("runs:/.../contact_predictor")

    # Predict from DataFrame
    df = pd.DataFrame([{'pdb_file': 'protein.pdb'}])
    results = served_model.predict(df)
"""

# Temporarily commented out due to syntax errors
# from .contact_predictor import (
#     ContactPredictor,
#     load_esm2_model_cached,
#     generate_esm2_embeddings_batch,
#     extract_sequence_from_pdb_simple,
#     generate_pattern_based_template_features,
#     assemble_68_channel_tensor
# )

from .pyfunc_model import (
    ContactPredictionPyFunc,
    create_pyfunc_model_instance,
    create_pyfunc_model,
    log_model_to_mlflow,
    create_pyfunc_model_from_checkpoint,
    load_pyfunc_model,
    predict_from_pdb_pyfunc,
    predict_batch_from_pdb,
    predict_from_sequence_pyfunc,
    validate_pyfunc_model,
    get_model_info,
    benchmark_model_performance
)

# Temporarily commented out due to dependency issues
# from .prediction_utils import (
#     predict_from_sequence_pyfunc,
#     validate_pyfunc_model,
#     get_model_info,
#     benchmark_model_performance
# )

__all__ = [
    'ContactPredictionPyFunc',
    'create_pyfunc_model_instance',
    'create_pyfunc_model',
    'log_model_to_mlflow',
    'load_pyfunc_model',
    'predict_from_pdb_pyfunc',
    'predict_batch_from_pdb',
    'predict_from_sequence_pyfunc',
    'validate_pyfunc_model',
    'get_model_info',
    'benchmark_model_performance',
    'create_pyfunc_model_from_checkpoint'
]

__version__ = '2.0.0'
__author__ = 'ESM2 Contact Prediction Team'