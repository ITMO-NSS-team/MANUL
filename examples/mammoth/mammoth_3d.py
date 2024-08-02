import ast
import numpy as np
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


def run_example():
    feature, target = form_dataset()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)

    base_individ = DataStructureGraph(data=train_features,
                                      cash_folder='mammoth_cash',
                                      n_neighbors=10,
                                      epsilon_neighborhood=0.18)

    base_individ.show_3d(labels=train_target, title='Before evolution')
    base_individ.show_2d(labels=train_target, euclidean=True)

    # raw model without graph
    base_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                         num_epochs=50,
                         batch_size=300,
                         problem='regres')
    base_model.train()
    base_train_loss = base_model.get_metric_on_train()
    base_test_loss = base_model.get_metric_on_test(test_features, test_target)

    # raw model with base individ graph
    with_graph_model = ModelNN(train_features, train_target,
                               num_epochs=50,
                               batch_size=300,
                               problem='regres')
    with_graph_model.train(base_individ, plot_convergence=True)
    with_graph_train_loss = with_graph_model.get_metric_on_train()
    with_graph_test_loss = base_model.get_metric_on_test(test_features, test_target)

    # check evolution convergence by 5 runs
    for i in range(5):
        with_evolution_model = ModelNN(train_features, train_target,
                                       num_epochs=50,
                                       batch_size=300,
                                       problem='regres')

        evolution = Evolution(base_individ=base_individ,
                              iterations=30,
                              population_size=7,
                              model_to_optimize=with_evolution_model,
                              base_mutation=True,
                              edges_mutation=True,
                              edges_weight_mutation=True)
        evolution.run()
        evolution.plot_evolution_fitnesses()

        evolution.base_individ.show_2d(train_target, euclidean=True)
        evolution.base_individ.show_3d(train_target, title=f'After evolution {i}')

        with_evolution_train_loss = with_evolution_model.get_metric_on_train()
        with_evolution_test_loss = with_evolution_model.get_metric_on_test(test_features, test_target)

        # plot on train set
        b1 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
        for b in b1:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on train set')
        plt.show()
        # plot on test set
        b2 = plt.bar(['base', 'with graph', 'with evolution'],
                     [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
        for b in b2:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
        plt.title('MSE on test set')
        plt.show()

run_example()
