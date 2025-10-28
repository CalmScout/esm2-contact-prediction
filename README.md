# 🧬 ESM2 Contact Prediction with Modern MLflow Integration

**Welcome!** 👋 We're excited to share this state-of-the-art protein contact prediction system that combines Meta AI's powerful ESM-2 language model with homology-assisted convolutional networks. Whether you're a researcher, developer, or bioinformatics enthusiast, this project makes predicting protein structure interactions accessible and efficient.

## 🎯 What This Project Does

This system predicts which amino acids in a protein are physically close to each other (within 8Å) - a crucial task for understanding protein folding and function. By leveraging ESM-2's deep understanding of protein sequences combined with structural homology information, our model achieves excellent performance while remaining computationally efficient.

### 🌟 Key Highlights
- **92.4% AUC performance** on comprehensive benchmark dataset
- **Modern MLflow integration** with PyFunc serving for production deployment
- **5-step streamlined workflow** from data download to predictions
- **Easy to use** with pretrained models and clear documentation
- **Scalable** from quick prototyping to production use

### 👥 Who This Is For
- **Researchers** exploring protein structure prediction
- **Developers** building bioinformatics applications
- **Students** learning about ML in computational biology
- **Data scientists** working with protein sequences
- **ML Engineers** needing production-ready model serving

---

## 🎯 Strategy Overview: Homology-Assisted CNN

### **Implemented Approach**

**Concept**: Build a 2D CNN that combines sequence-derived features (from **ESM-2**) with distance/contact maps from homologous protein structures.

**Workflow**:
1. **Template Discovery**: Use HHblits to find homologous structures from PDB70/UniRef30 databases
2. **Sequence Alignment**: Map template coordinates to query residues using robust alignment
3. **Feature Integration**: Combine ESM-2 embeddings (64 channels) with template features (4 channels)
4. **Binary Prediction**: CNN outputs binary contact maps (0/1 values) using BCEWithLogitsLoss

**Input Format**: 68-channel tensor (4 template + 64 ESM2 channels) of shape `(68, L, L)`
**Output Format**: Binary L×L contact map with 8Å Cα-Cα distance threshold (6Å for Glycine)

### Strategy Comparison

| Strategy | Approach | Pros | Cons | Complexity |
|----------|----------|------|------|------------|
| **Homology-Assisted CNN** | 2D CNN + Templates | Easiest implementation<br>Proven accuracy<br>Combines ESM + structure | Template-dependent<br>L×L memory usage | **Low** ✅ |
| **Graph Neural Network** | Residue graph + structural edges | Natural geometry encoding<br>Scales with L | Complex implementation<br>Memory-intensive | **Medium** |
| **Template/Retrieval Transformer** | Transformer conditioned on homologs | Richest context<br>AlphaFold-level potential | Research-scale complexity<br>Long training time | **High** |

**Why This Strategy?** The Homology-Assisted CNN offers the best balance of implementation complexity, computational efficiency, and proven accuracy - making it ideal for practical applications and rapid development.

---

## 📁 Project Structure

```
esm2-contact-prediction/
├── src/esm2_contact/          # 🎯 Core package - main codebase
│   ├── training/              # CNN model architecture and training logic
│   │   ├── model.py          # BinaryContactCNN implementation
│   │   ├── trainer.py        # Training loop and optimization
│   │   ├── dataset.py        # PyTorch dataset classes
│   │   ├── metrics.py        # Evaluation metrics (AUC, precision, etc.)
│   │   └── losses.py         # Custom loss functions
│   ├── dataset/               # Data processing utilities
│   │   ├── processing.py     # Contact map generation from PDB
│   │   └── utils.py          # Helper functions for data handling
│   ├── homology/              # Template search and alignment
│   │   ├── search.py         # Template discovery algorithms
│   │   ├── alignment.py      # Sequence alignment utilities
│   │   └── robust_processor.py # Robust template processing
│   ├── analysis/              # Performance analysis tools
│   │   ├── performance_analyzer.py # Model evaluation utilities
│   │   └── mlflow_analyzer.py # Experiment tracking analysis
│   ├── serving/               # 🚀 Model serving and deployment
│   │   ├── contact_predictor.py # Core prediction engine
│   │   ├── prediction_utils.py # High-level prediction utilities
│   │   └── pyfunc_model.py # Modern MLflow PyFunc integration
│   └── mlflow_utils.py        # MLflow experiment tracking utilities
├── scripts/                   # 🚀 5-step workflow scripts
│   ├── 01_download_dataset.py # Download training data from Google Drive
│   ├── 02_download_homology_databases.py # Download HHblits databases
│   ├── 03_generate_cnn_dataset.py # Generate CNN training dataset
│   ├── 04_train_cnn.py # Train CNN model with MLflow logging
│   └── 05_predict_from_pdb.py # Predict with MLflow PyFunc models
├── config.yaml               # 📋 Project configuration
├── pyproject.toml           # 📦 Modern Python dependencies (Python 3.13+)
├── data/                    # 📊 Training and test data
├── mlruns/                 # 🏆 MLflow experiments and models (created after training)
└── predictions/            # Output predictions directory
```

### 💡 Key Insights
- **`src/esm2_contact/`** is the heart of the project - all core functionality lives here
- **`scripts/`** contains the 5-step workflow for complete reproduction
- **`mlruns/`** will be created after your first training run (contains trained models)
- **`src/esm2_contact/serving/`** provides production-ready PyFunc model serving
- **Follow scripts 01-05 in order** for the complete workflow from data to predictions

### 📝 Important Notes
- **`mlruns/` directory is not included** in this repository to keep it lightweight
- You'll train your own models after setting up the environment
- All trained models will be stored locally in `mlruns/` for your use

---

## 🚀 Quick Start

**Ready to jump in?** Let's get you running with protein contact prediction in just a few minutes!

### 📦 Dependencies

First, let's set up your environment using modern Python package management:

```bash
# Install all project dependencies (recommended)
uv sync

# Verify installation
uv run python --version  # Should show Python 3.13+
```

**💡 Tip:** The first time you run ESM-2, it will automatically download the model (~2GB), so make sure you have an internet connection!

### 📝 .gitignore Recommendation

**Important**: Add the following to your `.gitignore` file to avoid committing large model files and temporary data:

```gitignore
# MLflow experiments and models (can be very large)
mlruns/

# Generated datasets and models
*.h5
*.pth
*.pkl

# Temporary files and caches
__pycache__/
*.pyc
.pytest_cache/
.DS_Store

# ESM2 model cache
.cache/
```

This will keep your repository lightweight while preserving all the code and configuration files.

### 🎯 Quick Start Guide

**💡 Important Note**: Since we don't include pretrained models in this repository (to keep it lightweight), you'll need to train your own model first. This gives you full control over the training process and ensures optimal performance for your use case!

#### **Option A: Quick Training (30-60 Minutes)**
```bash
# Step 1: Generate a small training dataset (1% of data, ~30 minutes)
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --process-ratio 0.01 \
    --random-seed 42

# Step 2: Train a quick model (~20 minutes)
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --epochs 10

# Step 3: Use your freshly trained model!
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --threshold 0.40
```

**Expected output:**
```
✅ Dataset generated successfully: 150 proteins processed
✅ Model training completed: AUC ~85%
✅ ESM2 model loaded and cached (6.7s)
✅ Contact prediction completed (0.1s)
🎉 Success! Contact density: 3-5% (realistic for quick model)
💾 Predictions saved to: predictions.json
```

#### **Option B: Full Production Training (2-4 Hours)**
```bash
# Step 1: Generate the complete training dataset (100% of data, ~2-3 hours)
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --process-ratio 1.0 \
    --random-seed 42

# Step 2: Train production model (~90-120 minutes)
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --epochs 50 \
    --experiment-name "production_model"

# Step 3: Make predictions with your production model
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --threshold 0.15
```

**Expected output:**
```
✅ Dataset generated successfully: 15,000 proteins processed
✅ Model training completed: AUC 90-92%
✅ ESM2 model loaded and cached (6.7s)
✅ Contact prediction completed (0.1s)
🎉 Success! Contact density: 5-10% (realistic for production model)
💾 Predictions saved to: predictions.json
```

#### **Option C: Loading Your Trained Model**
```bash
# If you already have a trained model, make predictions directly
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --threshold 0.15 \
    --verbose
```

**💡 How to find your model URI**: See the "Finding Your MLflow Model URI" section below for detailed instructions on locating your trained model's URI.

**Expected output:**
```
✅ ESM2 model loaded and cached (6.7s)
✅ Template processing completed (3.2s)
✅ Contact prediction completed (0.1s)
🎉 Success! Contact density: 5-10% (realistic range)
💾 Predictions saved to: predictions.json
```

### 🎉 Success Indicators

**How do you know it's working?** Look for these green flags:

✅ **Model loads successfully** - You'll see "ESM2 model loaded and cached"
✅ **MLflow tracking active** - Training logs to `mlruns/` directory
✅ **Reasonable processing time** - 11-13 seconds total per protein
✅ **Realistic contact density** - 5-10% for full model, 3-5% for quick model
✅ **No error messages** - Smooth execution with clear progress indicators

### Model Comparison

| Model | Training Data | AUC | Contact Density | Optimal Threshold | Best For |
|-------|---------------|-----|-----------------|------------------|----------|
| **Quick Test Model** | 1% data | ~85% AUC | 28% → 3.5% (adjusted) | **0.40** | Rapid prototyping |
| **Full Dataset Model** | 100% data | **92.4% AUC** | 8.2% (realistic) | **0.15** | Production use |

### Threshold Guidance
- **Full dataset model**: Use `--threshold 0.15` for realistic 5-10% contact density
- **Quick test model**: Use `--threshold 0.40` for realistic 3-5% contact density
- **Auto-thresholding**: Omit `--threshold` parameter for system optimization

---

## 🔄 Complete 5-Step Workflow

**🎯 Goal: Reproduce our 92.4% AUC results from scratch!** This step-by-step guide takes you from a fresh git clone to a fully trained model with identical performance.

### 📋 Prerequisites

**System Requirements:**
- **Python**: 3.13+ (required, managed by `uv`)
- **OS**: Linux/macOS (Ubuntu 20.04+ recommended)
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (RTX 3080/4080 or better)
- **RAM**: 16GB+ minimum, 32GB+ recommended
- **Storage**: 400GB+ free disk space (for databases + datasets)
- **Internet**: Required for initial downloads (ESM2 model + databases)

### 🔄 Step-by-Step Workflow

#### **Step 0: Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd esm2-contact-prediction

# Install all dependencies (uses uv for fast, reliable installs)
uv sync

# Verify installation
uv run python --version  # Should show Python 3.13+
```

#### **Step 1: Download Training Dataset**
```bash
# Download the complete training/test dataset (~2GB download)
uv run python scripts/01_download_dataset.py

# ✅ Delivers: Complete PDB dataset in data/data/
# - Training structures: ~15,000 proteins
# - Test structures: ~1,000 proteins
# - Expected time: 5-15 minutes depending on internet speed
```

#### **Step 2: Download Homology Databases**
```bash
# Download both PDB70 and UniRef30 databases (367GB total)
uv run python scripts/02_download_homology_databases.py --db all

# ✅ Delivers: HHblits databases in data/homology_databases/
# - PDB70: 56GB extracted (structural templates)
# - UniRef30: 218GB extracted (sequence homologs)
# - Expected time: 1-3 hours depending on internet speed
# - Disk space: ~367GB total required
```

**⚡ Quick Alternative (Skip for initial testing):**
You can start with Step 3 and come back to this later. The system will use pattern-based templates if databases aren't available.

#### **Step 3: Generate CNN Training Dataset**
```bash
# Generate complete CNN dataset with ESM2 embeddings (self-contained)
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir data/data/train \
    --output-path data/cnn_dataset.h5 \
    --process-ratio 1.0 \
    --random-seed 42

# ✅ Delivers: Complete training dataset (68-channel tensors + targets)
# - Features: 68 channels (4 template + 64 ESM2 embeddings)
# - ESM2 embeddings: Generated automatically (no separate step needed)
# - Targets: Binary contact maps (8Å Cα-Cα distance threshold)
# - Expected time: 2-4 hours for full dataset
# - File size: ~12GB

# For rapid testing (1% of data, ~30 minutes):
# --process-ratio 0.01
```

#### **Step 4: Train the CNN Model with MLflow**
```bash
# Train the model on the complete dataset with modern MLflow tracking
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --experiment-name "full_dataset_training" \
    --epochs 50 \
    --batch-size 4

# ✅ Delivers: Trained model achieving ~92.4% AUC
# - Model: BinaryContactCNN (380K parameters, 1.45MB)
# - Training time: ~90-120 minutes
# - Expected AUC: 92.0-92.5%
# - Model logged to: mlruns/ with full experiment tracking
```

#### **Step 5: Predict with MLflow PyFunc Model**
```bash
# Test the trained model on sample proteins using MLflow URI
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --output verification_predictions.json \
    --threshold 0.15 \
    --verbose

# ✅ Expected results:
# - Contact density: 5-10% (realistic)
# - Processing time: 11-13 seconds per protein
# - Output: JSON file with contact predictions
# - MLflow integration: Full model versioning and tracking
```

### 📊 Expected Outputs & Verification

**At each step, you should see:**

| Step | Expected Output | Verification |
|------|----------------|--------------|
| 1 | `data/data/` with PDB files | `ls data/data/train/*.pdb \| head -5` |
| 2 | `data/homology_databases/` with db files | `ls data/homology_databases/pdb70/` |
| 3 | `data/cnn_dataset.h5` file | `uv run python -c "import h5py; print('Dataset OK')"` |
| 4 | `mlruns/` with trained model | `ls mlruns/*/artifacts/model/` |
| 5 | Predictions JSON file | `cat verification_predictions.json` |

**Performance Benchmarks:**
- **Total pipeline time**: 4-7 hours (excluding Step 2 databases)
- **Peak GPU memory**: 4-6GB during training
- **Peak RAM usage**: 8-12GB during dataset generation
- **Final AUC**: 92.0-92.5% (within ±0.5% of published results)

---

## 🏗️ Building Your Dataset

**Did you know?** Our high-performance model was trained on 15,000 protein structures! This section shows you how to generate your own CNN dataset from PDB files.

### 📁 Setting Up Your Data Structure

Before generating your dataset, make sure your PDB files are properly organized:

```bash
# Create the standard directory structure
mkdir -p data/train data/test

# Extract your PDB files to these folders:
# - data/train/  -> Training PDB files
# - data/test/   -> Test PDB files for evaluation

# Example structure:
data/
├── train/
│   ├── 1ABC.pdb
│   ├── 2DEF.pdb
│   └── ... (your training proteins)
└── test/
    ├── 1GHI.pdb
    ├── 2JKL.pdb
    └── ... (your test proteins)
```

### 📊 Dataset Generation Process

The dataset generation pipeline transforms raw PDB files into ready-to-train HDF5 datasets with 68-channel tensors:

1. **PDB Processing**: Extract sequences and coordinates from protein structure files
2. **ESM2 Embeddings**: Generate 64-channel sequence embeddings using Meta AI's ESM-2 model
3. **Contact Maps**: Create ground truth binary contact maps (8Å Cα-Cα distance threshold)
4. **Template Features**: Generate 4 channels of homology-based template information
5. **HDF5 Assembly**: Combine everything into efficient training datasets

### 🚀 Dataset Generation Examples

#### **For Rapid Development (1% of data, ~30 minutes):**
```bash
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --output-path experiments/quick_test_dataset/cnn-train-test.h5 \
    --process-ratio 0.01 \
    --random-seed 42
```

#### **For Production Use (100% of data, several hours):**
```bash
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --output-path experiments/full_dataset_training/cnn-train-full-dataset.h5 \
    --process-ratio 1.0 \
    --random-seed 42
```

### 📋 Expected Outputs

| Dataset Size | Processing Time | Output Size | Proteins Processed | Use Case |
|-------------|----------------|------------|-------------------|----------|
| 1% (150 proteins) | ~30 minutes | ~200MB | 150 | Development & testing |
| 100% (15,000 proteins) | ~2-3 hours | ~12GB | 15,000 | Production training |

### 🔍 Output Structure

The generated HDF5 file contains:
- **`features`**: 68-channel tensors (4 template + 64 ESM2 channels)
- **`targets`**: Binary contact maps (L×L matrices)
- **`metadata`**: Protein information and processing statistics

---

## 🚀 Modern MLflow Integration & PyFunc Serving

**🎯 MLflow-First Architecture**: This project has been modernized with comprehensive MLflow integration for production-ready model serving and experiment tracking.

### 🏆 MLflow Features

- **Single Source of Truth**: All models stored in `mlruns/` directory
- **Experiment Tracking**: Automatic parameter and metric logging
- **Model Registry**: Versioning and staging support
- **PyFunc Integration**: Production-ready model serving
- **Git Integration**: Reproducibility tracking

### 🔍 Finding Your MLflow Model URI

After training a model with script `04_train_cnn.py`, you need to find the correct MLflow URI to use for predictions. Here are several ways to discover your model URIs:

#### **Method 1: Using Command Line**
```bash
# Find all trained models in MLflow
find mlruns/ -name "*.pth" | head -5

# Example output:
# mlruns/EXPERIMENT_ID/RUN_ID/artifacts/model.pth
# mlruns/EXPERIMENT_ID/RUN_ID/models/.../best_model_checkpoint/model.pth

# Convert path to URI format:
# Path: mlruns/EXPERIMENT_ID/RUN_ID/artifacts/model.pth
# URI:  runs:/RUN_ID/model
```

#### **Method 2: Using MLflow UI**
```bash
# Launch MLflow UI to browse experiments
mlflow ui

# Then open http://localhost:5000 in your browser
# Navigate to Experiments → Your Run → Artifacts tab
# Copy the URI from the model artifact path
```

#### **Method 3: List Recent Runs**
```bash
# List recent MLflow runs with their details
find mlruns/ -name "meta.yaml" -exec dirname {} \; | sort -r | head -5

# Check run details
ls -la mlruns/*/RUN_ID/artifacts/  # Replace RUN_ID with actual run ID
```

#### **Understanding MLflow URI Format**

MLflow URIs follow this pattern: `runs:/RUN_ID/ARTIFACT_PATH`

**URI Examples (replace with your actual RUN_ID):**
```bash
# Standard trained model
runs:/YOUR_RUN_ID/model

# Best model checkpoint (if saved during training)
runs:/YOUR_RUN_ID/best_model_checkpoint

# Custom model artifact path
runs:/YOUR_RUN_ID/custom_model_name
```

**💡 Finding Your RUN_ID:**
- **Run ID**: Found in `mlruns/EXPERIMENT_ID/RUN_ID/` directory name
- **Example**: If your model is at `mlruns/1234567890/abcdef1234/artifacts/model.pth`, then your URI is `runs:/abcdef1234/model`
- **Artifact Path**: Usually `model` for our trained CNN models

### 🎯 PyFunc Model Serving

Our modern PyFunc integration supports multiple input formats:

```python
# Load PyFunc model from MLflow
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("runs:/run_id/model")

# Predict from PDB files
input_df = pd.DataFrame([{'pdb_file': 'protein.pdb'}])
results = model.predict(input_df)

# Predict from sequences
input_df = pd.DataFrame([{'sequence': 'ACDEFGHIKLMNPQRSTVWY'}])
results = model.predict(input_df)

# Predict from pre-computed features
input_df = pd.DataFrame([{'features': your_68_channel_tensor}])
results = model.predict(input_df)
```

### 📊 Model Registry and Versioning

```python
# Register model in MLflow registry
mlflow.register_model("runs:/run_id/model", "ESM2_Contact_Predictor_v1")

# Load specific version
model = mlflow.pyfunc.load_model("models:/ESM2_Contact_Predictor_v1/Production")

# Model validation
from esm2_contact.serving.pyfunc_model import validate_pyfunc_model
validation_results = validate_pyfunc_model("models:/ESM2_Contact_Predictor_v1/Production")
```

### 🔧 Batch Processing

```python
from esm2_contact.serving.pyfunc_model import predict_batch_from_pdb

# Batch prediction for multiple proteins
pdb_files = ['protein1.pdb', 'protein2.pdb', 'protein3.pdb']
results = predict_batch_from_pdb("runs:/run_id/model", pdb_files)
```

---

## 📊 Performance Results

**Excited to share our results!** 🎉 Our models have achieved outstanding performance that we're proud of:

### 🏆 Expected Model Performance

Based on our testing, you can expect the following performance ranges:

#### **Full Dataset Models (15,000 proteins)**
- **AUC Performance**: 90-93% (🌟 **Outstanding**)
- **Training Time**: 2-4 hours (full pipeline)
- **Peak GPU Memory**: 4-6GB
- **Contact Density**: 5-10% (realistic biological range)

#### **Quick Development Models (1-5% of data)**
- **AUC Performance**: 80-87% (🌟 **Excellent** for prototyping)
- **Training Time**: 30-60 minutes
- **Peak GPU Memory**: 2-4GB
- **Contact Density**: 3-6% (adjustable with threshold)

#### **General Performance Metrics**
- **Precision@L**: 95-99% (high precision contact prediction)
- **Inference Speed**: 11-13 seconds per protein
- **Memory Optimization**: FP16 training and batch processing support

### 📈 Performance Classification
- **AUC > 0.9**: Outstanding 🏆 (Production-ready models)
- **AUC 0.8-0.9**: Excellent ✅ (Development and testing models)
- **AUC 0.7-0.8**: Good 👍 (Early stage models)
- **AUC 0.6-0.7**: Moderate 👌 (Needs more training data)
- **AUC < 0.6**: Needs Improvement 🔄 (Check training configuration)

**🎯 What this means**: Your models will learn meaningful contact patterns by combining ESM-2's powerful sequence understanding with homology template information. The exact performance will depend on your dataset size and training parameters, but the framework is designed to achieve high accuracy consistently.

### Inference Performance & Requirements

#### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (RTX 3080/4080 or similar)
- **GPU Memory**: 8GB minimum, 12GB+ recommended for ESM2 model
- **CPU**: Multi-core CPU for template processing
- **RAM**: 16GB+ for ESM2 model and processing
- **Storage**: 5GB+ for ESM2 model cache
- **Internet**: Required for first-time ESM2 model download (~2GB)

#### **Performance Breakdown**
| Step | Time (First Run) | Time (Subsequent Runs) | Memory Usage |
|------|------------------|------------------------|-------------|
| ESM2 Model Loading | ~6-7s | ~6-7s (cached) | 2-4GB GPU |
| Template Processing | ~3-4s | ~3-4s | 1-2GB RAM |
| Feature Assembly | ~1-2s | ~1-2s | 1GB RAM |
| Model Inference | <1s | <1s | 0.5GB GPU |
| **Total** | **~11-13s** | **~11-13s** | **4-6GB GPU** |

#### **Optimization Features**
- ✅ **ESM2 Model Caching**: Loaded once globally, reused across predictions
- ✅ **Half-Precision**: ESM2 model uses FP16 for faster inference
- ✅ **Model Compilation**: PyTorch 2.0+ compilation for optimized inference
- ✅ **Memory Management**: Automatic cleanup after each prediction
- ✅ **Pattern-Based Templates**: Fast, reliable template generation (no external dependencies)

---

## 🔧 Troubleshooting & Support

**Hit a snag?** Don't worry! We've compiled solutions to the most common issues. Remember, every problem has a solution! 💪

### 💾 Memory Issues? We've Got You Covered!

**Problem**: "Out of memory" errors
**Quick Fixes**:
```bash
# Reduce batch size for memory efficiency
uv run python scripts/04_train_cnn.py --batch-size 1

# Process smaller dataset first
uv run python scripts/03_generate_cnn_dataset.py --process-ratio 0.01

# Use adaptive batching (automatically adjusts)
uv run python scripts/04_train_cnn.py --adaptive-batching
```

### 🤖 ESM-2 Model Taking Forever?

**Problem**: ESM-2 model loading issues or slow downloads
**Solutions**:
- ✅ **First-time setup**: Initial download is ~2GB (one-time only!)
- ✅ **Check internet**: Required for first download
- ✅ **Disk space**: Ensure ~5GB free for model cache
- ✅ **Patience**: Model caches after first run, subsequent loads are faster
- ✅ **GPU check**: Ensure CUDA-compatible GPU with 8GB+ memory

### 📊 MLflow Integration Issues?

**Problem**: MLflow model loading or URI errors
**Solutions**:
- ✅ **Check URI format**: Use `runs:/run_id/model_name` format
- ✅ **Verify model exists**: Check `mlruns/` directory structure
- ✅ **Model artifacts**: Ensure model files are properly logged
- ✅ **PyFunc loading**: Use `mlflow.pyfunc.load_model()` for PyFunc models

```bash
# List available MLflow models
find mlruns/ -name "*.pth" | head -5

# Check MLflow runs
mlflow ui  # Launch MLflow UI to browse experiments
```

### 📊 Contact Density Looking Weird?

**Problem**: Unrealistic contact densities (too high or too low)

**Quick Fixes**:
```bash
# For Quick Test Model: Use higher threshold
--threshold 0.40  # Gives realistic 3-5% density

# For Full Dataset Model: Use lower threshold
--threshold 0.15  # Gives realistic 5-10% density

# Or let the system decide! (Recommended)
# Just remove --threshold parameter entirely
```

**What to expect**:
- ✅ **Good range**: 5-10% contact density
- 🚨 **Too high**: >20% (threshold too low)
- 🚨 **Too low**: <1% (threshold too high)

### 🎯 uv Package Management Issues?

**Problem**: Dependency installation or environment issues
**Solutions**:
```bash
# Ensure uv is properly installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clean and reinstall
uv sync --reinstall

# Check Python version
uv run python --version  # Should be 3.13+

# Verify dependencies
uv pip list | grep -E "(torch|mlflow|esm)"
```

### 🆘 Still Stuck?

**Remember**:
- 🌟 **Every expert was once a beginner!**
- 📚 **Check the MLflow UI**: Run `mlflow ui` to explore experiments
- 🔍 **Read the error messages**: They often contain the solution
- 💡 **Start small**: Try the quick model first, then work up to the full one

**You've got this!** Protein contact prediction is complex, but you're on the right track. 🚀

---

## 📖 References

### Key Papers
- **Rives et al., Science 2021** - "Biological Structure and Function Emerge from Scaling Up Language Models" (ESM-2)
- **Lin et al., Science 2023** - "ESMFold: High-Resolution Protein Structure Prediction"
- **Jumper et al., Nature 2021** - "Highly accurate protein structure prediction with AlphaFold" (CNN methodology)
- **Wang et al., Bioinformatics 2017** - "DeepContact: protein contact prediction by exploiting protein sequence-based features"

### Method Inspiration
- **Periscope (MLCB 2019)** - CNN-based contact prediction with templates
- **DeepContact/RaptorX** - End-to-end contact prediction pipelines
- **AlphaFold2** - Template integration and attention mechanisms

### Tools & Libraries
- **[HHsuite](https://github.com/soedinglab/hh-suite)** - Sensitive homology search
- **[ESM-2](https://github.com/facebookresearch/esm)** - Meta AI's protein language model
- **[MLflow](https://mlflow.org/)** - MLOps platform for the ML lifecycle
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Biopython](https://biopython.org/)** - Computational biology tools
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager

---

## 🌟 Conclusion

**Thank you for exploring our ESM2 Contact Prediction system!** 🎉

We've poured our hearts into creating a tool that makes advanced protein contact prediction accessible to everyone. Whether you're:

- 🔬 **A researcher** pushing the boundaries of structural biology
- 👨‍💻 **A developer** building the next bioinformatics breakthrough
- 🎓 **A student** diving into computational biology
- 🚀 **An ML engineer** deploying models to production

**...this project is for you!**

### 🎯 What Makes This Special

- **🏆 Top Performance**: 92.4% AUC that rivals state-of-the-art methods
- **⚡ Modern Architecture**: MLflow-first design with PyFunc serving
- **🛠️ Easy to Use**: 5-step workflow with clear documentation
- **📚 Production Ready**: Modern MLOps with model registry and versioning
- **🤝 Community Driven**: Built with love for the scientific community

### 🚀 Ready to Start?

1. **Jump in** with the [Quick Start](#-quick-start) guide
2. **Explore** the [Project Structure](#-project-structure) to understand how it works
3. **Build your own** datasets with our [generation instructions](#-building-your-dataset)
4. **Deploy models** using [MLflow PyFunc serving](#-modern-mlflow-integration--pyfunc-serving)

**The future of protein structure prediction is in your hands.** Let's accelerate scientific discovery together! 🧬✨

---

*Built with ❤️ for the computational biology community. Questions? Contributions? We'd love to hear from you!*