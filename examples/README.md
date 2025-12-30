# MANUL Examples: Gradient Isomap Pipeline

This directory contains complete examples demonstrating the MANUL pipeline for manifold learning and graph regularization.

## Overview

The pipeline consists of **two stages** that work together:

1. **Stage 1: Manifold Learning** - Learns the intrinsic geometry of data using GradientIsomap
2. **Stage 2: Graph Regularization** - Trains neural networks with graph-based regularization using the learned manifold

Data flows between stages through the **`outputs/`** directory within each experiment folder.

---

## Directory Structure

```
MANUL/
├── examples/
│   └── gradient_isomap/
│       ├── mnist/                     # MNIST classification example
│       │   ├── outputs/               # Experiment outputs (timestamped runs)
│       │   │   └── run_20251210_150023_n2000/
│       │   │       ├── fps_indices.npy
│       │   │       ├── best_distance_matrix.npy
│       │   │       ├── base_projections.npy
│       │   │       ├── train_projections.npy
│       │   │       ├── experiment_metadata.json
│       │   │       ├── hist_plot.png              # Dimension histogram
│       │   │       ├── variance_plot.png          # Variance analysis
│       │   │       └── experiment_20251210_145530/  # Stage 2 results
│       │   │           ├── baseline/
│       │   │           │   └── best_model.pth
│       │   │           ├── regularized/
│       │   │           │   ├── best_model.pth
│       │   │           │   └── *_convergence.png
│       │   │           ├── mnist_comparison.png
│       │   │           ├── experiment_config.json
│       │   │           └── comparison_results.csv
│       │   └── stages/
│       │       ├── first_stage.py     # Stage 1: Manifold Learning
│       │       └── second_stage.py    # Stage 2: Graph Regularization
│       └── synthetic/                 # Synthetic geometry regression example
│           ├── outputs/               # Experiment outputs
│           │   └── torus_run_20251211_192332_n10000/
│           │       ├── fps_indices.npy
│           │       ├── best_distance_matrix.npy
│           │       ├── base_projections.npy
│           │       ├── train_projections.npy
│           │       ├── experiment_metadata.json
│           │       └── experiment_20251211_193045/
│           │           ├── baseline/best_model.pth
│           │           ├── regularized/
│           │           │   ├── best_model.pth
│           │           │   └── *_convergence.png
│           │           ├── torus_comparison.png
│           │           ├── experiment_config.json
│           │           └── comparison_results.csv
│           └── stages/
│               ├── first_stage.py
│               └── second_stage.py
└── data/                              # Raw datasets (shared across experiments)
    └── MNIST/                         # Downloaded MNIST data
```

---

## Pipeline Workflow

### Stage 1: Manifold Learning

**Purpose:** Learn the intrinsic low-dimensional structure of high-dimensional data

**Process:**
1. Load or generate dataset
2. Split into train/val/test (70%/15%/15% for synthetic, stratified 64%/16%/20% for MNIST)
3. Apply Farthest Point Sampling (FPS) to select representative base points from training set
4. Train GradientIsomap on base points to learn geodesic distances
5. Compute low-dimensional projections for all training data points
6. Save results to `outputs/{run_folder_name}/`

**Key Parameters (editable in `first_stage.py`):**
- **MNIST:** `n_samples=2000` (FPS base points), `latent_dim` (auto-detected), `epochs=15000`
- **Synthetic:** `n_samples=10000` (total points), `n_base_points=2000`, `noise_percent=0.05`, `latent_dim=2`, `epochs=500`

**Outputs saved to `outputs/{run_folder_name}/`:**
- `fps_indices.npy` - Selected base point indices
- `best_distance_matrix.npy` - Learned geodesic distance matrix (upper triangular)
- `base_projections.npy` - Low-dimensional projections of base points
- `train_projections.npy` - Projections for training data (computed via ensemble KNN/random forest)
- `experiment_metadata.json` - Configuration (includes `latent_dim`, `random_seed`, split params)
- Visualization plots (Isomap convergence, dimension analysis for MNIST)

**Note:** Data splits (X_train, y_train, etc.) are regenerated in Stage 2 from `experiment_metadata.json` using the same random seed, ensuring reproducibility.

### Stage 2: Graph Regularization Training

**Purpose:** Train neural networks with manifold-aware regularization

**Process:**
1. Load preprocessed data from `outputs/{run_folder_name}/`
2. Reconstruct full distance matrix from upper triangular matrix
3. Train **baseline model** (no graph regularization, λ_graph=0)
4. Train **regularized model** (with graph regularization, λ_graph>0)
5. Compare performance and generate visualizations
6. Save experiments to `outputs/{run_folder_name}/experiment_{timestamp}/`

**Key Training Features:**
- **Full-batch gradient descent** 
- **Early stopping** based on validation loss median
- **Adaptive lambda**: Sobol sensitivity analysis at 10% of training
- **Graph regularization loss:** Symmetric normalized Laplacian with RBF kernel

**Key Parameters (editable at bottom of `second_stage.py`):**
- **MNIST:** `reg_lambda=1`, `num_epochs=200`, `batch_size=128`, `learning_rate=1e-4`, `early_stopping_patience=150`
- **Synthetic:** `reg_lambda=1`, `num_epochs=20000`, `batch_size=1024`, `learning_rate=1e-2`, `early_stopping_patience=5000`
- **Adaptive lambda:** `adaptive_lambda='sobol'` or `False`

**Outputs saved to `outputs/{run_folder_name}/experiment_{timestamp}/`:**
- `baseline/best_model.pth` - Best baseline model weights
- `regularized/best_model.pth` - Best regularized model weights
- `{dataset}_comparison.png` - Training/validation loss curves, accuracy plots (MNIST)
- `experiment_config.json` - Full experiment configuration with lambda history
- `comparison_results.csv` - Performance metrics (accuracy/MSE, R², improvement %)

---

## Usage

### Option 1: Run Stages Manually 

**Step 1: Run Stage 1 (Manifold Learning)**

```bash
cd examples/gradient_isomap/mnist/stages  # or synthetic/stages
python first_stage.py
```

This will create a timestamped folder like `outputs/run_20251210_150023_n2000/` (MNIST) or `outputs/torus_run_20251211_192332_n10000/` (synthetic).

**Step 2: Configure Stage 2**

Open `second_stage.py` and set the run folder name at the top:

```python
# Set the name of your run folder from Stage 1
RUN_FOLDER_NAME = 'run_20251210_150023_n2000'  # MNIST example
# or
RUN_FOLDER_NAME = 'torus_run_20251211_192332_n10000'  # Synthetic example
```

**Step 3: Run Stage 2 (Graph Regularization)**

```bash
python second_stage.py
```

This will automatically:
- Load data from the specified run folder
- Train baseline and regularized models
- Save results to `outputs/{run_folder}/experiment_{timestamp}/`

### Option 2: Run Full Pipeline (Automated)

```bash
# For MNIST classification
cd examples/gradient_isomap/mnist
python run_pipeline.py  

# For synthetic geometries
cd examples/gradient_isomap/synthetic
python run_pipeline.py 
```

**Note:** Pipeline scripts automatically pass folder names between stages.

---
