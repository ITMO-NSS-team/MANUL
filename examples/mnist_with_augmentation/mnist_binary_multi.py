from datetime import datetime
from copy import deepcopy

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evolution.Evolution import MultiEvolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN

def get_data():
    features = np.load("examples/data/feature_mnist.npy")
    target = np.load("examples/data/target_mnist.npy")
    angles = np.load("examples/data/angle_mnist.npy")
    # data is already shuffled for class balance
    new_features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))
    new_feature = []
    new_target = []
    new_angles = []
    for i, elem in enumerate(target):
        if elem in [0, 1]:
            # adding only two kind of numbers
            new_feature.append(new_features[i])
            new_target.append(elem)
            new_angles.append(angles[i])
    samples_num = 20000
    new_feature = np.array(new_feature[:samples_num], dtype='int64')
    new_target = np.array(new_target[:samples_num])
    new_angles = np.array(new_angles[:samples_num])
    return new_feature, new_target, new_angles


def split_dataset(data, split_ratio=0.8):
    split_ratio = int(data.shape[0] * split_ratio)
    train = data[:split_ratio]
    test = data[split_ratio:]
    return train, test

def form_markers_by_angle(angles):
    angles = angles.astype(object)
    angles[angles == 0] = 'circle'
    angles[angles == 15] = 'circle-open'
    angles[angles == 45] = 'x'
    angles[angles == 75] = 'cross'
    angles[angles == 105] = 'diamond'
    angles[angles == 135] = 'diamond-open'
    angles[angles == 165] = 'square'
    return angles


def run_example(mut):
    pop_size = 10
    iterations = 5
    if mut:
        nam = ''
    else:
        nam = 'noweightmut'
    f_folder = f'cache_mnist_n_runs/{nam}_{iterations}_{pop_size}_mnist_2class'

    feature, target, angles = get_data()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)
    train_angles, test_angles = split_dataset(angles)

    train_angles = form_markers_by_angle(train_angles)
    test_angles = form_markers_by_angle(test_angles)

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
                                        n_neighbors=20)

    base_model = ModelNN(train_features, train_target,
                            num_epochs=50,
                            batch_size=300,
                            problem='binary_class')
    base_model.train()
    base_train_loss = base_model.get_metric_on_train()
    base_test_loss = base_model.get_metric_on_test(test_features, test_target)

    with_graph_model = ModelNN(train_features, train_target,
                                num_epochs=50,
                                batch_size=300,
                                problem='binary_class')
    with_graph_model.train(base_individ)
    with_graph_train_loss = with_graph_model.get_metric_on_train()
    with_graph_test_loss = with_graph_model.get_metric_on_test(test_features, test_target)

    with_evolution_model = ModelNN(train_features, train_target,
                                    num_epochs=50,
                                    batch_size=300,
                                    problem='binary_class')

    evolution = MultiEvolution(base_individ=base_individ,
                            iterations=iterations,
                            population_size=pop_size,
                            model_to_optimize=with_evolution_model,
                            edges_weight_mutation=mut)
    
    evolution.run()

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
        plt.title('ROC AUC on train set')
        plt.savefig(f'{cache_folder}/results/individ_train{_}')
        plt.close()
        b2 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_test_loss, with_graph_test_loss, _test_loss])
        for b in b2:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('ROC AUC on test set')
        plt.savefig(f'{cache_folder}/results/individ_test{_}')
        plt.close()



if __name__ == "__main__":
    run_example(True)