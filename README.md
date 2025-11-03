# 🧬 ESM2 Contact Prediction with Modern MLflow Integration

**Welcome!** 👋 This state-of-the-art protein contact prediction system combines Meta AI's ESM-2 language model with homology-assisted convolutional networks for accurate protein structure interaction prediction.

## 🎯 What This Project Does

Predicts which amino acids in a protein are physically close (within 8Å) - crucial for understanding protein folding and function. By leveraging ESM-2's sequence understanding with structural homology information, the system achieves excellent performance while remaining computationally efficient.

### 🌟 Key Highlights
- **93.3% AUC performance** on comprehensive benchmark dataset
- **Modern MLflow integration** with PyFunc serving for production deployment
- **5-step streamlined workflow** from data download to predictions
- **Production-ready** with pretrained models and comprehensive documentation
- **Scalable** from prototyping to production use

---

## 🏗️ Architecture Overview

### **Strategy: Homology-Assisted CNN**

**Concept**: 2D CNN combining sequence-derived features (ESM-2) with distance/contact maps from homologous protein structures.

**Workflow**:
1. **Template Discovery**: Find homologous structures using HHblits
2. **Sequence Alignment**: Map template coordinates to query residues
3. **Feature Integration**: Combine ESM-2 embeddings (64 channels) with template features (4 channels)
4. **Binary Prediction**: CNN outputs binary contact maps using BCEWithLogitsLoss

**Input**: 68-channel tensor `(68, L, L)` - 4 template + 64 ESM2 channels
**Output**: Binary L×L contact map with 8Å Cα-Cα distance threshold

### 📊 Workflow Visualization

![ESM2 Contact Prediction Workflow](mermaid_chart.png)

---

## 📁 Project Structure

```
esm2-contact-prediction/
├── src/esm2_contact/          # 🎯 Core package
│   ├── training/              # CNN model architecture and training
│   ├── dataset/               # Data processing utilities
│   ├── homology/              # Template search and alignment
│   ├── analysis/              # Performance analysis tools
│   ├── serving/               # 🚀 Model serving and deployment
│   └── mlflow_utils.py        # MLflow experiment tracking
├── scripts/                   # 🚀 5-step workflow scripts
│   ├── 01_download_dataset.py
│   ├── 02_download_homology_databases.py
│   ├── 03_generate_cnn_dataset.py
│   ├── 04_train_cnn.py
│   └── 05_predict_from_pdb.py
├── notebooks/                 # 📓 Analysis notebooks
├── config.yaml               # 📋 Project configuration
├── pyproject.toml           # 📦 Modern Python dependencies
└── data/                    # 📊 Training and test data
```

**💡 Key Insights**:
- **`src/esm2_contact/`** contains all core functionality
- **`scripts/`** provides the complete 5-step workflow
- **Follow scripts 01-05 in order** for the full pipeline
- **`mlruns/`** is created after training (contains trained models)

---

## 🚀 Quick Start

### 📦 Environment Setup

```bash
# Install all project dependencies
uv sync

# Verify installation
uv run python --version  # Should show Python 3.13+
```

**💡 Tip**: First ESM-2 run downloads ~2GB model automatically.

### 🎯 Using Pre-trained Model (Recommended)

**Single Protein Prediction:**
```bash
uv run python scripts/05_predict_from_pdb.py \
  --pdb-file data/data/test/4LCY.pdb \
  --model-path mlruns/329887396556374988/4988988931424d8c9dfcb42c3a54ec2b/artifacts/best_model_checkpoint/tmp4s837oa3.pth
```

**Batch Prediction:**
```bash
uv run python scripts/05_predict_from_pdb.py \
  --batch-processing \
  --pdb-file-dir data/data/test/ \
  --model-path mlruns/329887396556374988/4988988931424d8c9dfcb42c3a54ec2b/artifacts/best_model_checkpoint/tmp4s837oa3.pth \
  --output batch_predictions.json
```

**Expected Output:**
```
🔄 Loading model from file: mlruns/.../model.pth
   ✅ Model loaded successfully on cuda (380,033 parameters)
🎉 Prediction completed successfully!
   📄 Results: predictions.json
   📏 Sequence length: 766
   📊 Contact density: 5.58%
   🔢 Total contacts: 32,746
```

**💡 Model Specifications:**
- **Performance**: 93.3% AUC
- **Architecture**: Efficient 32-base channel CNN
- **Parameters**: 380,033
- **Processing Time**: ~10 seconds per protein

### 🏗️ Training Your Own Model

**Quick Training (30-60 minutes):**
```bash
# Step 1: Generate small dataset (1% of data)
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --process-ratio 0.01 \
    --random-seed 42

# Step 2: Train model
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --epochs 10

# Step 3: Make predictions
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --threshold 0.40
```

**Production Training (2-4 hours):**
```bash
# Generate full dataset
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir ./data/data/train \
    --process-ratio 1.0 \
    --random-seed 42

# Train production model
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --epochs 50 \
    --experiment-name "production_model"
```

---

## 🔄 Complete 5-Step Workflow

**Goal**: Reproduce 93.3% AUC results from scratch

### 📋 Prerequisites
- **Python**: 3.13+ (managed by `uv`)
- **OS**: Linux/macOS (Ubuntu 20.04+ recommended)
- **GPU**: CUDA-compatible with 8GB+ VRAM
- **RAM**: 16GB+ minimum, 32GB+ recommended
- **Storage**: 400GB+ free space
- **Internet**: Required for initial downloads

### 🔄 Step-by-Step

**Step 0: Environment Setup**
```bash
git clone <repository-url>
cd esm2-contact-prediction
uv sync
```

**Step 1: Download Training Dataset**
```bash
uv run python scripts/01_download_dataset.py
# ✅ Delivers: ~15,000 training + ~1,000 test proteins (5-15 min)
```

**Step 2: Download Homology Databases**
```bash
uv run python scripts/02_download_homology_databases.py --db all
# ✅ Delivers: PDB70 + UniRef30 databases (367GB total, 1-3 hours)
```

**Step 3: Generate CNN Training Dataset**
```bash
uv run python scripts/03_generate_cnn_dataset.py \
    --pdb-dir data/data/train \
    --output-path data/cnn_dataset.h5 \
    --process-ratio 1.0 \
    --random-seed 42
# ✅ Delivers: 68-channel tensors + targets (2-4 hours, ~12GB)
```

**Step 4: Train CNN Model**
```bash
uv run python scripts/04_train_cnn.py \
    --dataset-path data/cnn_dataset.h5 \
    --experiment-name "full_dataset_training" \
    --epochs 50 \
    --batch-size 4
# ✅ Delivers: Trained model ~93.3% AUC (90-120 min)
```

**Step 5: Predict with MLflow PyFunc**
```bash
uv run python scripts/05_predict_from_pdb.py \
    --pdb-file data/data/test/1BB3.pdb \
    --model-uri "runs:/YOUR_RUN_ID/model" \
    --threshold 0.15 \
    --verbose
# ✅ Expected: 5-10% contact density, 11-13 seconds processing
```

---

## 🏆 Performance Results

### Expected Model Performance

| Dataset Size | AUC Performance | Training Time | Contact Density | Best For |
|--------------|----------------|---------------|-----------------|----------|
| **Full (15k proteins)** | **90-93%** 🏆 | 2-4 hours | 5-10% | Production |
| **Quick (1-5% data)** | **80-87%** ✅ | 30-60 min | 3-6% | Development |

### Performance Metrics
- **Precision@L**: 95-99%
- **Inference Speed**: 11-13 seconds per protein
- **Peak GPU Memory**: 4-6GB during training
- **Model Size**: 380K parameters (1.45MB)

### Hardware Requirements
- **GPU**: CUDA-compatible (RTX 3080/4080+ recommended)
- **GPU Memory**: 8GB minimum, 12GB+ recommended
- **RAM**: 16GB+ for ESM2 model and processing
- **Storage**: 5GB+ for ESM2 model cache

---

## 🚀 MLflow Integration & PyFunc Serving

### Finding Your Model URI

After training, find your model using:
```bash
# List available models
find mlruns/ -name "*.pth" | head -5

# Launch MLflow UI
mlflow ui  # Open http://localhost:5000
```

**URI Format**: `runs:/RUN_ID/model_name`

### PyFunc Model Usage

```python
import mlflow.pyfunc
import pandas as pd

# Load model
model = mlflow.pyfunc.load_model("runs:/run_id/model")

# Predict from PDB files
input_df = pd.DataFrame([{'pdb_file': 'protein.pdb'}])
results = model.predict(input_df)

# Batch processing
from esm2_contact.serving.pyfunc_model import predict_batch_from_pdb
pdb_files = ['protein1.pdb', 'protein2.pdb']
results = predict_batch_from_pdb("runs:/run_id/model", pdb_files)
```

---

## 🔧 Troubleshooting

### Common Issues

**Memory Issues:**
```bash
# Reduce batch size
uv run python scripts/04_train_cnn.py --batch-size 1

# Process smaller dataset
uv run python scripts/03_generate_cnn_dataset.py --process-ratio 0.01
```

**ESM-2 Loading Issues:**
- First-time download: ~2GB (one-time only)
- Ensure internet connection for initial download
- Check disk space (~5GB free required)
- Verify CUDA-compatible GPU

**Contact Density Issues:**
```bash
# Quick model: Use higher threshold
--threshold 0.40  # 3-5% density

# Production model: Use lower threshold
--threshold 0.15  # 5-10% density

# Auto-threshold (recommended)
# Omit --threshold parameter entirely
```

**MLflow URI Issues:**
```bash
# Check URI format: runs:/run_id/model
# Verify model exists in mlruns/ directory
# Use mlflow ui to browse experiments
```

### Success Indicators
✅ Model loads successfully
✅ Reasonable processing time (11-13 seconds)
✅ Realistic contact density (5-10% for production)
✅ No error messages

---

## 📖 References

### Key Papers
- **Rives et al., Science 2021** - ESM-2 protein language model
- **Lin et al., Science 2023** - ESMFold structure prediction
- **Jumper et al., Nature 2021** - AlphaFold2 methodology

### Tools & Libraries
- **[ESM-2](https://github.com/facebookresearch/esm)** - Meta AI's protein language model
- **[MLflow](https://mlflow.org/)** - MLOps platform for experiment tracking
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[HHsuite](https://github.com/soedinglab/hh-suite)** - Homology search tools
- **[uv](https://docs.astral.sh/uv/)** - Modern Python package manager

---

## 🌟 Conclusion

This project makes advanced protein contact prediction accessible with:

- 🏆 **Top Performance**: 93.3% AUC
- ⚡ **Modern Architecture**: MLflow-first design with PyFunc serving
- 🛠️ **Easy to Use**: 5-step workflow with clear documentation
- 📚 **Production Ready**: Complete MLOps integration

**Ready to start?** Jump into the [Quick Start](#-quick-start) and begin predicting protein contacts today! 🧬✨

---

*Built with ❤️ for the computational biology community.*