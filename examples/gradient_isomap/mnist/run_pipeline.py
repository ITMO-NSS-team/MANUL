"""
Full MNIST Classification Pipeline
===================================

Runs both stages sequentially:
  Stage 1: Manifold Learning with GradientIsomap
  Stage 2: Classification with Graph Regularization

Usage:
    python run_pipeline.py
"""
import subprocess
import sys
import os

def run_stage(stage_name, script_path):
    """Execute a pipeline stage script"""
    print(f"\n{'='*60}")
    print(f"Running {stage_name}")
    print(f"{'='*60}\n")

    result = subprocess.run([sys.executable, script_path])

    if result.returncode != 0:
        print(f"\nError: {stage_name} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n{stage_name} completed successfully")
    return result.returncode

if __name__ == "__main__":
    print("="*60)
    print("MNIST Classification Pipeline")
    print("="*60)
    print("This will run:")
    print("  Stage 1: Manifold Learning (Isomap)")
    print("  Stage 2: Graph Regularization (Classifier Training)")
    print()

    run_stage(
        stage_name="Stage 1: Manifold Learning",
        script_path="stages/01_manifold_learning.py"
    )

    run_stage(
        stage_name="Stage 2: Graph Regularization",
        script_path="stages/02_graph_regularization.py"
    )

    print("\n" + "="*60)
    print("Full pipeline completed successfully!")
    print("="*60)
