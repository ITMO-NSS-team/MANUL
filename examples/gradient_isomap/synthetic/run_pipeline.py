"""
Full Synthetic Geometry Pipeline
=================================

Runs both stages sequentially:
  Stage 1: Manifold Learning with GradientIsomap
  Stage 2: Regression with Graph Regularization
"""
import os
from datetime import datetime

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

    # Here output directory for all runs can be specified
    outputs_dir = 'outputs'

    geometries_to_process = [
        'torus',
        #'sphere',
        #'swiss_roll',
        #'swiss_hole',
        #'s_curve',
        #'pseudosphere',
        #'hyperboloid',
        #'helicoid',
        #'multi_scale_torus',
        #'nonuniform_sphere',
        #'cone_surface',
        #'genus_2_surface',
        #'connected_multiscale_manifold',
    ]

    for geom in geometries_to_process:

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        working_folder = os.path.join(outputs_dir, f'{geom}_run_{timestamp}')
        os.makedirs(working_folder, exist_ok=True)

        stage1_results_folder = synthetic_manifold_learning_pipeline(geom, working_folder)

        print(f"\nStage 1 completed successfully")

        print(f"\n{'='*60}")
        print("Running Stage 2: Graph Regularization")
        print(f"{'='*60}\n")

        stage2_results = synthetic_graph_regularization(folder_path=stage1_results_folder)
        print(f"\nStage 2 for {geom} completed successfully")

    print("\n" + "="*60)
    print("Full pipeline completed successfully!")
    print("="*60)
