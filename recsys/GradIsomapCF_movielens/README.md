# GradientIsomapCF: Geometry of Recommendation Feature Spaces with Differentiable Isomap

This repository contains code and experiments for the research project on **studying the geometry of item feature spaces in recommender systems** using a **differentiable Isomap** pipeline.

The main idea is to move beyond an *a priori* choice of geometry (Euclidean / hyperbolic) and instead **learn an item-item distance matrix** end-to-end and then **analyze the resulting manifold geometry**.

---
## What is inside

- **Pure NCF (NeuMF) baseline** for implicit feedback recommendation.
- **GradientIsomapCF (IsomapNN + NCF) - current version GradientIsomapCF_log.py**:
  - IsomapNN produces manifold coordinates \(Z\) from a learnable distance matrix (D_input).
  - NeuMFOnManifold (NCF) uses \(Z\) as item representations.
  - Training follows a **bilevel scheme**:
    - **Inner loop:** train NCF with fixed \(Z\)
    - **Outer step:** freeze NCF and update (D_input) by backpropagating the recommendation loss through IsomapNN

---

## Data

Experiments are run on **MovieLens-1M** in a **next-item prediction** setup (leave-one-out split with negative sampling).  
A structured subsample of **most active users** and **most popular items** is used to obtain a denser interaction graph, which makes neighborhood graphs and geodesic estimates more stable.

Place MovieLens-1M files under:


Dataset: https://grouplens.org/datasets/movielens/1m/

---

## How to run

The main entry point is the `main()` function in the experiment script - file main_new2.py.

You can enable:
- `run_ncf=True` to run the **baseline NCF**
- `run_gincf=True` to run **GradientIsomapCF**

---

## Outputs

Training logs, saved models, and plots are written to the `logs_*` folders (separate directories for baseline and GradientIsomapCF runs).  
For GradientIsomapCF, distance/embedding snapshots are saved per outer epoch for downstream geometric analysis.

---

## Metrics

Evaluation uses standard sampled ranking metrics:
- **HR@K**
- **NDCG@K**
