"""
Full Synthetic Geometry Pipeline
=================================

Runs both stages sequentially:
  Stage 1: Manifold Learning with GradientIsomap
  Stage 2: Regression with Graph Regularization
"""
import sys
import os

from stages.first_stage import synthetic_manifold_learning_pipeline
from stages.second_stage import synthetic_graph_regularization

if __name__ == "__main__":
    print("="*60)
    print("Synthetic Geometry Regression Pipeline")
    print("="*60)
    print("This will run:")
    print("  Stage 1: Manifold Learning (Isomap)")
    print("  Stage 2: Graph Regularization (Regressor Training)")
    print()

    print(f"\n{'='*60}")
    print("Running Stage 1: Manifold Learning")
    print(f"{'='*60}\n")

    stage1_results = synthetic_manifold_learning_pipeline()
    print(f"\nStage 1 completed successfully")

    print(f"\n{'='*60}")
    print("Running Stage 2: Graph Regularization")
    print(f"{'='*60}\n")

    if stage1_results and 'geometry_folders' in stage1_results:
        for geometry_name, folder_path in stage1_results['geometry_folders'].items():
            print(f"\n{'='*40}")
            print(f"Processing geometry: {geometry_name}")
            print(f"{'='*40}\n")

            stage2_results = synthetic_graph_regularization(folder_path=folder_path)
            print(f"\nStage 2 for {geometry_name} completed successfully")

    print("\n" + "="*60)
    print("Full pipeline completed successfully!")
    print("="*60)
