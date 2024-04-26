from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def get_data():
    features = np.load("../data/feature_mnist.npy")
    target = np.load("../data/target_mnist.npy")
    angles = np.load("../data/angle_mnist.npy")
    # data is already shuffled for class balance
    new_features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))
    new_feature = []
    new_target = []
    new_angles = []
    for i, elem in enumerate(target):
        if elem not in [9]:
            # remove 9 from classification because augmentation makes it equal to 6
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


def plot_mnist(with_weights_path, no_weights_path, save_path):
    weight_df = pd.read_csv(with_weights_path)
    weight_df = weight_df.drop(columns=['Unnamed: 0'])

    weight_df = weight_df[['test_base', 'test_with_graph', 'test_with_evolution']]
    weight_df = weight_df.rename({
        'test_base': 'Test\nNo graph',
        'test_with_graph': 'Test\nInitial graph',
        'test_with_evolution': 'Test\nEvolution graph'}, axis='columns')

    no_weight_df = pd.read_csv(no_weights_path)
    no_weight_df = no_weight_df.drop(columns=['Unnamed: 0'])

    no_weight_df = no_weight_df[['test_base', 'test_with_graph', 'test_with_evolution']]
    no_weight_df = no_weight_df.rename({
        'test_base': 'Test\nNo graph',
        'test_with_graph': 'Test\nInitial graph',
        'test_with_evolution': 'Test\nEvolution graph'}, axis='columns')

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    weight_df.boxplot(showfliers=False, ax=ax[0])
    no_weight_df.boxplot(showfliers=False, ax=ax[1])

    ax[0].set_ylabel('F1 score')
    ax[0].set_ylim(0, 0.7)
    ax[1].set_ylim(0, 0.7)
    ax[0].set_title('With geometry mutation')
    ax[1].set_title('Without geometry mutation')
    plt.suptitle('MNIST augmentation dataset (8 class)\nF1 score by 10 runs')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def run_example(n_runs, mut):
    pop_size = 5
    iterations = 50
    if mut:
        nam = ''
    else:
        nam = 'noweightmut'

    f_folder = f'mnist_n_runs_results/{nam}_{iterations}_{pop_size}_mnist_8class'

    feature, target, angles = get_data()
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
        cash_folder = f'{f_folder}/{start_time}'
        base_individ = DataStructureGraph(data=train_features,
                                          cash_folder=cash_folder,
                                          n_neighbors=20
                                          )

        base_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                             num_epochs=200,
                             batch_size=300, problem='multiclass')
        base_model.train()
        base_train_loss = base_model.get_metric_on_train()
        base_test_loss = base_model.get_metric_on_test(test_features, test_target)

        with_graph_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                                   num_epochs=200,
                                   batch_size=300, problem='multiclass')
        with_graph_model.train(base_individ)
        with_graph_train_loss = with_graph_model.get_metric_on_train()
        with_graph_test_loss = with_graph_model.get_metric_on_test(test_features, test_target)

        with_evolution_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                                       num_epochs=200,
                                       batch_size=300, problem='multiclass')

        evolution = Evolution(base_individ=base_individ,
                              iterations=iterations,
                              population_size=pop_size,
                              model_to_optimize=with_evolution_model,
                              edges_weight_mutation=True)
        evolution.run()
        evolution.plot_evolution_fitnesses()
        evolution.base_individ.show_2d(train_target, save_path=f'{cash_folder}/final_graph.png')
        evolution.plot_evolution_fitnesses(save_path=f'{cash_folder}/evolution_conv.png')

        with_evolution_train_loss = with_evolution_model.get_metric_on_train()
        with_evolution_test_loss = with_evolution_model.get_metric_on_test(test_features, test_target)

        b1 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
        for b in b1:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('F1 on train set')
        plt.show()
        b2 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
        for b in b2:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('F1 on test set')
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
        df.to_csv(f'{f_folder}/{n_runs}_mnist_8_class_aug.csv')


run_example(n_runs=10, mut=True)
run_example(n_runs=10, mut=False)
plot_mnist('mnist_n_runs_results/_50_5_mnist_8class/10_mnist_8_class_aug.csv',
           'mnist_n_runs_results/noweightmut_50_5_mnist_8class/10_mnist_8_class_aug.csv',
           'mnist_n_runs_results/with_without_weights_mutation_comparison.png')
