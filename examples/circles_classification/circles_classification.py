import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def generate_dataset():
    np.random.seed()
    # Step 1: Generate the dataset
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)

    # Convert to PyTorch tensors
    '''X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)'''

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    '''plt.scatter(X_train[:, 1], X_train[:, 0], c=y_train)
    plt.title('Train')
    plt.show()

    plt.scatter(X_test[:, 1], X_test[:, 0], c=y_test)
    plt.title('Test')
    plt.show()'''
    return X_train, X_test, y_train, y_test


train_features, test_features, train_target, test_target = generate_dataset()

iters = 500
model_epochs = 300
folder = f'circles_class(one_crit)_{iters}'
if not os.path.exists(folder):
    os.makedirs(folder)

# TEST WITHOUT GRAPH
base_model = ModelNN(train_features, train_target,
                     num_epochs=model_epochs,
                     batch_size=300,
                     problem='binary_class')
base_model.train(plot_convergence=True)
with_graph_train_loss = base_model.get_metric_on_train()
with_graph_test_loss = base_model.get_metric_on_test(test_features, test_target)
output = base_model.model(torch.tensor(test_features).to('cuda'))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[0].set_title('Target classes')
axs[1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
axs[1].set_title('Predicted classes')
fig.suptitle(f'Raw model: Train ROC AUC={with_graph_train_loss}, Test ROC AUC={with_graph_test_loss}')
plt.tight_layout()
plt.savefig(f'{folder}/raw_model_prediction.png')
plt.show()


# TEST WITH INITIAL GRAPH
base_individ = DataStructureGraph(data=train_features,
                                  cache_folder=folder,
                                  n_neighbors=10,
                                  epsilon_neighborhood=0.18,
                                  data_labels=train_target,
                                  fully_connected=True)
base_individ.show_2d(save_path=f'{folder}/initial_graph.png')
with_graph_model = ModelNN(train_features, train_target,
                           num_epochs=model_epochs,
                           batch_size=300,
                           problem='binary_class')
with_graph_model.train(base_individ, plot_convergence=True)
with_graph_train_loss = with_graph_model.get_metric_on_train()
reproj_test_features = base_individ.isomap_data_projection(test_features)
with_graph_test_loss = base_model.get_metric_on_test(reproj_test_features, test_target)

output = with_graph_model.model(torch.tensor(test_features).to('cuda'))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=test_target)
axs[0, 0].set_title('Reprojected target classes')
axs[0, 1].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=output.cpu().detach().numpy())
axs[0, 1].set_title('Reprojected predicted classes')
axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[1, 0].set_title('Euclidean target classes')
axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
axs[1, 1].set_title('Euclidean predicted classes')
fig.suptitle(f'Base graph model: Train ROC AUC={with_graph_train_loss}, Test ROC AUC={with_graph_test_loss}')
plt.tight_layout()
plt.savefig(f'{folder}/base_graph_prediction.png')
plt.show()


# TEST WITH EVOLUTION
with_evolution_model = ModelNN(train_features, train_target,
                               num_epochs=model_epochs,
                               batch_size=300,
                               problem='binary_class')

evolution = Evolution(base_individ=base_individ,
                      iterations=iters,
                      population_size=7,
                      model_to_optimize=with_evolution_model,
                      base_mutation=False,
                      edges_mutation=True,
                      edges_weight_mutation=True)
evolution.run(multicriteria=False)
#evolution.plot_evolution_pareto_fronts()
#evolution.plot_pareto(f'{folder}/evolution_res_pareto.png')
evolution.plot_evolution_fitnesses(reverse=True, save_path=f'{folder}/convergence.png')
#result_graphs = evolution.pareto
#result_models = evolution.pareto_models()

'''for g, graph in enumerate(result_graphs):
    graph.show_2d(save_path=f'{folder}/pareto_{g}.png', labels=train_target)
    model = result_models[g]
    reproj_test_features = graph.isomap_data_projection(test_features)
    output = model.model(torch.tensor(test_features).to('cuda'))

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=test_target)
    axs[0, 0].set_title('Reprojected target classes')
    axs[0, 1].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=output.cpu().detach().numpy())
    axs[0, 1].set_title('Reprojected predicted classes')
    axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
    axs[1, 0].set_title('Euclidean target classes')
    axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
    axs[1, 1].set_title('Euclidean predicted classes')
    fig.suptitle(f'{g} - evolution graph model: Train ROC AUC={with_graph_train_loss}, Test ROC AUC={with_graph_test_loss}')
    plt.tight_layout()
    plt.savefig(f'{folder}/{g}_graph_prediction.png')
    plt.show()'''

best_individ = evolution.best_individ
best_individ.show_2d(save_path=f'{folder}/best_individ.png', labels=train_target)
model = evolution.best_individ_model()
reproj_test_features = best_individ.isomap_data_projection(test_features)
output = model.model(torch.tensor(test_features).to('cuda'))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=test_target)
axs[0, 0].set_title('Reprojected target classes')
axs[0, 1].scatter(reproj_test_features[:, 1], reproj_test_features[:, 0], c=output.cpu().detach().numpy())
axs[0, 1].set_title('Reprojected predicted classes')
axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
axs[1, 0].set_title('Euclidean target classes')
axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
axs[1, 1].set_title('Euclidean predicted classes')
fig.suptitle(f'Best graph - evolution graph model: Train ROC AUC={with_graph_train_loss}, Test ROC AUC={with_graph_test_loss}')
plt.tight_layout()
plt.savefig(f'{folder}/best_graph_prediction.png')
plt.show()

print('Save final population')
for i, ind in enumerate(evolution.population.individs_pool):
    ind.show_2d(save_path=f'{folder}/{i}_last_generation.png')
