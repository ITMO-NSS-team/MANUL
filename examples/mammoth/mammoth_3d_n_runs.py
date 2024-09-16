import ast
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


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


def plot_mammoth(with_weights_path, no_weights_path, save_path):
    weight_df = pd.read_csv(with_weights_path)
    weight_df = weight_df.drop(columns=['Unnamed: 0'])

    weight_df = weight_df[['test_base', 'test_with_graph', 'test_with_evolution']]
    weight_df = weight_df.rename({
        'test_base': 'No graph',
        'test_with_graph': 'Initial graph',
        'test_with_evolution': 'Evolution graph'}, axis='columns')

    no_weight_df = pd.read_csv(no_weights_path)
    no_weight_df = no_weight_df.drop(columns=['Unnamed: 0'])

    no_weight_test_df = no_weight_df[['test_base', 'test_with_graph', 'test_with_evolution']]
    no_weight_test_df = no_weight_test_df.rename({
        'test_base': 'No graph',
        'test_with_graph': 'Initial graph',
        'test_with_evolution': 'Evolution graph'}, axis='columns')

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    weight_df.boxplot(showfliers=False, ax=ax[0])
    no_weight_test_df.boxplot(showfliers=False, ax=ax[1])

    ax[0].set_ylim(0, 0.008)
    ax[1].set_ylim(0, 0.008)
    ax[0].set_title('With geometry mutation')
    ax[1].set_title('Without geometry mutation')
    ax[0].set_ylabel('Mean absolute error')
    plt.suptitle('"Mammoth" dataset\nMAE by 10 runs')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def run_example(n_runs, mut):
    pop_size = 5
    iterations = 100
    if mut:
        nam = ''
    else:
        nam = 'noweightmut'
    f_folder = f'mammoth_n_runs_results_new/{nam}_{iterations}_{pop_size}'

    feature, target = form_dataset()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)

    train_base = []
    train_with_graph = []
    train_with_evolution = []
    test_base = []
    test_with_graph = []
    test_with_evolution = []

    for run in range(n_runs):
        start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S_%p')
        cach_folder = f'{f_folder}/{start_time}'

        base_individ = DataStructureGraph(data=train_features,
                                          cach_folder=cach_folder,
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

        evolution = Evolution(base_individ=base_individ,
                              iterations=iterations,
                              population_size=pop_size,
                              model_to_optimize=with_evolution_model,
                              edges_weight_mutation=mut)
        evolution.run()
        evolution.plot_evolution_fitnesses()
        evolution.base_individ.show_2d(train_target, save_path=f'{cach_folder}/final_graph.png')
        evolution.plot_evolution_fitnesses(save_path=f'{cach_folder}/evolution_conv.png')

        with_evolution_train_loss = with_evolution_model.get_metric_on_train()
        with_evolution_test_loss = with_evolution_model.get_metric_on_test(test_features, test_target)

        b1 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
        for b in b1:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on train set')
        plt.show()
        b2 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
        for b in b2:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on test set')
        plt.show()

        train_base.append(base_train_loss)
        train_with_graph.append(with_graph_train_loss)
        train_with_evolution.append(with_evolution_train_loss)

        test_base.append(base_test_loss)
        test_with_graph.append(with_graph_test_loss)
        test_with_evolution.append(with_evolution_test_loss)

        df = pd.DataFrame()
        df['train_base'] = train_base
        df['train_with_graph'] = train_with_graph
        df['train_with_evolution'] = train_with_evolution
        df['test_base'] = test_base
        df['test_with_graph'] = test_with_graph
        df['test_with_evolution'] = test_with_evolution
        df.to_csv(
            f'{f_folder}/{n_runs}_mammoth.csv')


run_example(10, True)
# run_example(10, False)
# plot_mammoth('mammoth_n_runs_results/_50_5/10_mammoth.csv',
#              'mammoth_n_runs_results/noweightmut_50_5/10_mammoth.csv',
#              'mammoth_n_runs_results/with_without_weights_mutation_comparison.png')
