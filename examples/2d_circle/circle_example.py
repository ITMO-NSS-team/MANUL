import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def get_data():
    train = pd.read_csv('data/circle_curve_train.csv')
    train_features = train[['x', 'y']]
    train_target = train['color']
    test = pd.read_csv('data/circle_curve_test.csv')
    test_features = test[['x', 'y']]
    test_target = test['color']
    return (np.array(train_features),
            np.array(train_target),
            np.array(test_features),
            np.array(test_target))


def run_example():
    iters = 10
    folder = f'circle_curve_(one_crit)_{iters}'
    train_features, train_target, test_features, test_target = get_data()
    base_individ = DataStructureGraph(data=train_features,
                                      cache_folder=folder,
                                      n_neighbors=10,
                                      epsilon_neighborhood=0.18,
                                      data_labels=train_target)

    base_individ.show_2d(save_path=f'{folder}/initial_graph.png')
    base_model = ModelNN(train_features, train_target,
                         num_epochs=150,
                         batch_size=300,
                         problem='regres',)
    base_model.train(plot_convergence=True)

    output = base_model.model(torch.tensor(test_features).to('cuda'))
    plt.scatter(test_features[:, 1], test_features[:, 0], c=test_target)
    plt.title('model test - True')
    plt.show()
    plt.scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
    plt.title('model test - Predicted')
    plt.show()

    base_train_loss = base_model.get_metric_on_train()
    base_test_loss = base_model.get_metric_on_test(test_features, test_target)

    # raw model with base individ graph
    with_graph_model = ModelNN(train_features, train_target,
                               num_epochs=150,
                               batch_size=300,
                               problem='regres')
    with_graph_model.train(base_individ, plot_convergence=True)
    with_graph_train_loss = with_graph_model.get_metric_on_train()
    reproj_test_features = base_individ.isomap_data_projection(test_features)
    with_graph_test_loss = base_model.get_metric_on_test(reproj_test_features, test_target)

    output = with_graph_model.model(torch.tensor(test_features).to('cuda'))
    plt.scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=output.cpu().detach().numpy())
    plt.title('model test - Predicted')
    plt.show()
    plt.scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
    plt.title('model test - Predicted')
    plt.show()

    # check evolution convergence by 5 runs
    for i in range(1):
        with_evolution_model = ModelNN(train_features, train_target,
                                       num_epochs=150,
                                       batch_size=300,
                                       problem='regres')

        evolution = Evolution(base_individ=base_individ,
                              iterations=iters,
                              population_size=7,
                              model_to_optimize=with_evolution_model,
                              base_mutation=False,
                              edges_mutation=True,
                              edges_weight_mutation=True)
        evolution.run()
        evolution.plot_evolution_pareto_fronts()
        evolution.plot_pareto(f'{folder}/evolution_res_pareto.png')
        result_graphs = evolution.pareto

        for g, graph in enumerate(result_graphs):
            graph.show_2d(save_path=f'{folder}/pareto_{g}.png', labels=train_target)

        result_models = evolution.pareto_models()
        for m, model in enumerate(result_models):
            graph = evolution.pareto[m]
            with_evolution_train_loss = model.get_metric_on_train()
            reproj_test_features = graph.isomap_data_projection(test_features)
            with_evolution_test_loss = model.get_metric_on_test(reproj_test_features, test_target)

            '''plt.scatter(test_features[:, 1], test_features[:, 0], c=test_target)
            plt.title('model test - True')
            plt.show()'''
            output = model.model(torch.tensor(test_features).to('cuda'))
            plt.scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
            plt.title('model test - Predicted')
            plt.show()
            plt.scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=output.cpu().detach().numpy())
            plt.title('model test - Predicted')
            plt.show()

            # plot on train set
            b1 = plt.bar(['base', 'with graph', 'with evolution'],
                         [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
            for b in b1:
                height = b.get_height()
                plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
            plt.title(f'mse on train set - {m} individ')
            plt.savefig(f'{folder}/metric_train_{m}.png')
            plt.show()
            # plot on test set
            b2 = plt.bar(['base', 'with graph', 'with evolution'],
                         [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
            for b in b2:
                height = b.get_height()
                plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
            plt.title(f'mse on test set - {m} individ')
            plt.savefig(f'{folder}/metric_test_{m}.png')
            plt.show()

run_example()