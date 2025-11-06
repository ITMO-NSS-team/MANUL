# MANUL Examples: Gradient Isomap Pipeline

This directory contains complete examples demonstrating the MANUL pipeline for manifold learning and graph regularization.

## Overview

The pipeline consists of **two stages** that work together:

1. **Stage 1: Manifold Learning** - Learns the intrinsic geometry of data using GradientIsomap
2. **Stage 2: Graph Regularization** - Trains neural networks with graph-based regularization using the learned manifold

Data flows between stages through the **`outputs/`** directory at the project root.

---

## Directory Structure

```
MANUL-main/
├── examples/
│   └── gradient_isomap/
│       ├── mnist/                     # MNIST classification example
│       │   ├── run_pipeline.py        # Full pipeline runner
│       │   └── stages/
│       │       ├── 01_manifold_learning.py
│       │       └── 02_graph_regularization.py
│       └── synthetic/                 # Synthetic geometry regression example
│           ├── run_pipeline.py        # Full pipeline runner
│           └── stages/
│               ├── 01_manifold_learning.py
│               └── 02_graph_regularization.py
├── outputs/                           # Stage 1 → Stage 2 data transfer
│   ├── mnist_2000/                    # MNIST with 2000 samples
│   └── torus/                         # Torus geometry
└── data/                              # Raw datasets
    └── MNIST/                         # Downloaded MNIST data
```

---

## Pipeline Workflow

### Stage 1: Manifold Learning

**Purpose:** Learn the intrinsic low-dimensional structure of high-dimensional data

**Process:**
1. Load or generate dataset
2. Apply Farthest Point Sampling (FPS) to select representative basis points
3. Train GradientIsomap to learn geodesic distances on the manifold
4. Compute low-dimensional projections for all data points
5. Save results to `outputs/{dataset_name}/`

**Outputs saved to `outputs/{dataset_name}/`:**
- `fps_indices.npy` - Selected basis point indices
- `best_distance_matrix.npy` - Learned geodesic distance matrix
- `base_projections.npy` - Low-dimensional projections of basis points
- `train_projections.npy` - Projections for training data
- `val_projections.npy` - Projections for validation data
- `X_train.npy, X_val.npy, X_test.npy` - Dataset splits (features)
- `y_train.npy, y_val.npy, y_test.npy` - Dataset splits (targets)
- `latent_dim.npy` - Estimated intrinsic dimensionality

### Stage 2: Graph Regularization Training

**Purpose:** Train neural networks with manifold-aware regularization

**Process:**
1. Load preprocessed data from `outputs/{dataset_name}/`
2. Train baseline model (no regularization, λ=0)
3. Train regularized model (with graph regularization, λ>0)
4. Compare performance and visualize results
5. Save experiments to `outputs/{dataset_name}/experiment_{timestamp}/`

**Outputs saved to `outputs/{dataset_name}/experiment_{timestamp}/`:**
- `baseline/` - Baseline model checkpoints and metrics
- `regularized/` - Regularized model checkpoints and metrics
- `mnist_comparison.png` or `{geometry}_comparison.png` - Visualization
- `experiment_config.json` - Full experiment configuration
- `comparison_results.npy` - Performance metrics

---

## Usage

### Option 1: Run Full Pipeline 


Automatically runs both stages sequentially:

```bash
# For MNIST classification
python examples/gradient_isomap/mnist/run_pipeline.py

# For synthetic geometries (torus, sphere, etc.)
python examples/gradient_isomap/synthetic/run_pipeline.py
```


### Option 2: Run Stages Manually

Run stages independently

```bash
# Stage 1: Manifold Learning
cd examples/gradient_isomap/mnist/stages
python 01_manifold_learning.py

# Stage 2: Graph Regularization (after Stage 1 completes)
python 02_graph_regularization.py
```

**Or from project root:**
```bash
python examples/gradient_isomap/mnist/stages/01_manifold_learning.py
python examples/gradient_isomap/mnist/stages/02_graph_regularization.py
```




---

## Examples

### 1. MNIST Classification

**Dataset:** MNIST handwritten digits (28×28 grayscale images)

**Task:** Multi-class classification (10 classes)

**Configuration:**
- **Stage 1:** 2000 FPS basis points, ~15,000 GradientIsomap epochs
- **Stage 2:**  MLP, λ=0 (baseline) vs λ=0.000001 (regularized)

**Run:**
```bash
python examples/gradient_isomap/mnist/run_pipeline.py
```


### 2. Synthetic Geometries (Torus, Sphere, etc.)

**Dataset:** Synthetically generated manifolds with noise

**Task:** Regression (predict color/coordinate values)

**Supported Geometries:**
- `torus` - 2D torus embedded in 3D
- `sphere` - 2D sphere surface
- `swiss_roll` - Classic manifold learning benchmark
- `s_curve` - S-shaped 2D manifold

**Configuration:**
- **Stage 1:** 5000 points with 5% noise, 1000 FPS basis points, 10,000 epochs
- **Stage 2:**  MLP, λ=0 vs λ=0.00001

**Run:**
```bash
python examples/gradient_isomap/synthetic/run_pipeline.py
```

**Customize geometry** (edit `01_manifold_learning.py` line 225):
```python
geometries_to_process = ['torus']  # Change to 'sphere', 'swiss_roll', etc.
```

---

