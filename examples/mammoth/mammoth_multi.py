import ast
import os
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evolution.Evolution import MultiEvolution
from evolution.IndividStructures import DataStructureGraph
from structure_approximation.ModuleNN import ModelNN


def form_dataset():
    """
    Load points and generate colors for mammoth dataset
    :return: ndarray with points coordinates, ndarray with colors from 0 to 1
    """
    fl = open("../data/mammoth_3d.json ", "r")
    data = fl.read()
    data = np.array(ast.literal_eval(data))
    colors = np.linspace(0, 0.9, len(data))
    data = np.array(sorted(data, key=lambda parameters: parameters[1]))
    new_data = []
    new_colors = []
    for i, dt in enumerate(data):
        new_data.append(dt)
        new_colors.append(colors[i])
    data = []
    colors = []
    temp_data = []
    temp_colors = []
    for i, dat in enumerate(new_data):
        if i % 2 != 0:
            temp_data.append(dat)
            temp_colors.append(new_colors[i])
        else:
            data.append(dat)
            colors.append(new_colors[i])
    colors.extend(temp_colors)
    data.extend(temp_data)
    return np.array(data), np.array(colors)


def split_dataset(data, split_ratio=0.8):
    split_ratio = int(data.shape[0] * split_ratio)
    train = data[:split_ratio]
    test = data[split_ratio:]
    return train, test

def run_example(mut):
    pop_size = 10
    iterations = 15
    if mut:
        nam = ''
    else:
        nam = 'noweightmut'
    f_folder = f'mammoth_multi_results/{nam}_{iterations}_{pop_size}'

    feature, target = form_dataset()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)

    train_base = []
    train_with_graph = []
    train_with_evolution = []
    test_base = []
    test_with_graph = []
    test_with_evolution = []

    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')
    cache_folder = f'{f_folder}/{start_time}'

    base_individ = DataStructureGraph(data=train_features,
                                        cache_folder=cache_folder,
                                        n_neighbors=10,
                                        epsilon_neighborhood=0.18, )

    base_model = ModelNN(train_features, train_target,
                            num_epochs=50,
                            batch_size=300,
                            problem='regres')
    base_model.train()
    base_train_loss = base_model.get_metric_on_train()
    base_test_loss = base_model.get_metric_on_test(test_features, test_target)

    with_graph_model = ModelNN(train_features, train_target,
                                num_epochs=50,
                                batch_size=300,
                                problem='regres')
    with_graph_model.train(base_individ)
    with_graph_train_loss = with_graph_model.get_metric_on_train()
    with_graph_test_loss = with_graph_model.get_metric_on_test(test_features, test_target)

    with_evolution_model = ModelNN(train_features, train_target,
                                    num_epochs=50,
                                    batch_size=300,
                                    problem='regres')

    evolution = MultiEvolution(base_individ=base_individ,
                            iterations=iterations,
                            population_size=pop_size,
                            model_to_optimize=with_evolution_model,
                            edges_weight_mutation=mut)
    
    evolution.run()
    evolution.plot_mse_best_individ_by_epoch(test_features, test_target)


    os.mkdir(f"{cache_folder}/results")
    for _, individ in enumerate(evolution.population.individs_pool):
        temp_model = deepcopy(evolution.base_model)
        temp_model.train(individ)
        individ.show_2d(train_target, save_path=f'{cache_folder}/results/individ_graph{_}')
        _train_loss = temp_model.get_metric_on_train()
        _test_loss = temp_model.get_metric_on_test(test_features, test_target)
        b1 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_train_loss, with_graph_train_loss, _train_loss])
        for b in b1:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on train set')
        plt.savefig(f'{cache_folder}/results/individ_train{_}')
        plt.close()
        b2 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_test_loss, with_graph_test_loss, _test_loss])
        for b in b2:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on test set')
        plt.savefig(f'{cache_folder}/results/individ_test{_}')
        plt.close()



if __name__ == "__main__":
    run_example(True)