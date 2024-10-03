import os
import ast

import numpy as np
import pandas as pd
from copy import deepcopy
from mammoth_3d_n_runs import form_dataset, split_dataset

from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationEvoOperators import Population
from regularizator.ModuleNN import ModelNN

def plot_loss(info_loss):
    import plotly.express as px

    fig = px.line(info_loss, x="iteration", y=info_loss.columns)

    fig.update_xaxes(
    dtick="M1",
    tickformat="%b\n%Y")
    fig.show()


if __name__ == "__main__":
    feature, target = form_dataset()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)

    f_folder = "mammoth_n_runs_results_new/_5_5" # path to directory with results

    for directory in os.listdir(f_folder):
        path_to_file = f"{f_folder}/{directory}/best_individs_by_iterations.pkl" # name of file with best individs
        if not os.path.isfile(path_to_file): continue
        instance_graph = DataStructureGraph(data=train_features, 
                                            cache_folder=f"{f_folder}/{directory}",
                                            graph_file='base_graph.pkl')

        pop_inidivids = Population(size=1, base_individ=instance_graph)
        pop_inidivids.load_individs_pool(path_to_file)

        losses = {
            "iteration": [],
            "model": [],
            "graph": [],
            "loss": [],
            "fitness": [],
            "num_edges": []
        }

        for i, individ in enumerate(pop_inidivids.individs_pool):

            loss_all = individ.trained_loss_values
            losses["iteration"].append(i)
            losses["model"].append(loss_all['model_loss'])
            losses["graph"].append(loss_all['graph_loss'])
            losses["loss"].append(loss_all['combined_loss'])
            losses["fitness"].append(individ.fitness)
            losses['num_edges'].append(individ.number_of_edges)

        df = pd.DataFrame(losses)
        plot_loss(df)