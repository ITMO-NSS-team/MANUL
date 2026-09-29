"""
Full MNIST Classification Pipeline
===================================

Runs both stages sequentially:
  Stage 1: Manifold Learning with GradientIsomap
  Stage 2: Classification with Graph Regularization
"""
import sys
import os

from stages.first_stage import mnist_manifold_learning_example
from stages.second_stage import mnist_graph_regularization

if __name__ == "__main__":
    print("="*60)
    print("MNIST Classification Pipeline")
    print("="*60)
    print("This will run:")
    print("  Stage 1: Manifold Learning (Isomap)")
    print("  Stage 2: Graph Regularization (Classifier Training)")
    print()

    print(f"\n{'='*60}")
    print("Running Stage 1: Manifold Learning")
    print(f"{'='*60}\n")

    stage1_results = mnist_manifold_learning_example()
    print(f"\nStage 1 completed successfully")

    print(f"\n{'='*60}")
    print("Running Stage 2: Graph Regularization")
    print(f"{'='*60}\n")

    stage2_results = mnist_graph_regularization(
        folder_path=stage1_results['working_folder']
    )
    print(f"\nStage 2 completed successfully")

    print("\n" + "="*60)
    print("Full pipeline completed successfully!")
    print("="*60)
