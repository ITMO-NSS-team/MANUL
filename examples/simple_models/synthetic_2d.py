import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleSimple import ModelSimple


def generate_ds():
    #  regular grids
    x = np.arange(0, 1000)
    y = np.arange(0, 1000)
    xs, ys = np.meshgrid(x, y)
    colors_image = np.sqrt(xs ** 2 + ys ** 2)

    # train set
    np.random.seed(0)
    x = np.random.randint(1000, size=300)
    y = np.random.randint(1000, size=300)
    train_colors = np.sqrt(x ** 2 + y ** 2)
    train_features = [np.ravel(x), np.ravel(y)]

    plt.imshow(colors_image)
    plt.scatter(train_features[0], train_features[1])
    plt.colorbar()
    plt.title('Train points')
    plt.show()

    # test set
    np.random.seed(1)
    x = np.random.randint(1000, size=100, )
    y = np.random.randint(1000, size=100)
    test_colors = np.sqrt(x ** 2 + y ** 2)
    test_features = [np.ravel(x), np.ravel(y)]

    plt.imshow(colors_image)
    plt.scatter(test_features[0], test_features[1])
    plt.colorbar()
    plt.title('Test points')
    plt.show()
    return colors_image, np.array(train_features).T.astype(float), np.array(train_colors).astype(float), np.array(
        test_features).T.astype(float), np.array(test_colors).astype(float)


colors_image, train_features, train_colors, test_features, test_colors = generate_ds()

base_individ = DataStructureGraph(data=train_features,
                                  n_neighbors=10,
                                  cash_folder='sin_test')

base_individ.show_2d(train_colors, cmap_name='Blues', euclidean=True)

base_model = ModelSimple(train_features[base_individ.basis], train_colors[base_individ.basis],
                     problem='regres')
base_model.train()
base_train_loss = base_model.get_metric_on_train()
base_test_loss = base_model.get_metric_on_test(test_features, test_colors)

with_graph_model = ModelSimple(train_features[base_individ.basis], train_colors[base_individ.basis], problem='regres')
with_graph_model.train(base_individ)
with_graph_train_loss = with_graph_model.get_metric_on_train()
with_graph_test_loss = with_graph_model.get_metric_on_test(test_features, test_colors)

with_evolution_model = ModelSimple(train_features[base_individ.basis], train_colors[base_individ.basis],
                               problem='regres')

evolution = Evolution(base_individ=base_individ,
                      iterations=200,
                      population_size=10,
                      model_to_optimize=with_evolution_model
                      )
evolution.run()

with_evolution_train_loss = with_evolution_model.get_metric_on_train()
with_evolution_test_loss = with_evolution_model.get_metric_on_test(test_features, test_colors)

b1 = plt.bar(['base', 'with graph', 'with evolution'], [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
for b in b1:
    height = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')
plt.title('MSE on train set')
plt.show()
b2 = plt.bar(['base', 'with graph', 'with evolution'], [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
for b in b2:
    height = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2.0, height, f'{height:.5f}', ha='center', va='bottom')

plt.title('MSE on test set')
plt.show()

evolution.plot_evolution_fitnesses()
evolution.base_individ.show_2d(labels=train_colors, euclidean=True)
evolution.base_individ.show_2d(labels=train_colors, euclidean=False)
