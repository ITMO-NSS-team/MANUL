import numpy as np
from matplotlib import pyplot as plt

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


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
    x = np.random.randint(1000, size=100)
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

base_model = ModelNN(train_feature=train_features[base_individ.basis],
                     train_target=train_colors[base_individ.basis],
                     batch_size=10,
                     problem='regres')
base_model.train(plot_convergence=True, num_epochs=50)
base_train_loss = base_model.get_metric_on_train()
base_test_loss = base_model.get_metric_on_test(test_features, test_colors)

with_graph_model = ModelNN(train_features[base_individ.basis], train_colors[base_individ.basis],
                           batch_size=10, problem='regres')
with_graph_model.train(graph=base_individ, num_epochs=50, plot_convergence=True)
with_graph_train_loss = with_graph_model.get_metric_on_train()
with_graph_test_loss = with_graph_model.get_metric_on_test(test_features, test_colors)

with_evolution_model = ModelNN(train_features[base_individ.basis], train_colors[base_individ.basis],
                               batch_size=10, problem='regres', num_epochs=50)

evolution = Evolution(base_individ=base_individ,
                      iterations=30,
                      population_size=10,
                      model_to_optimize=with_evolution_model
                      )
evolution.run()

with_evolution_train_loss = with_evolution_model.get_metric_on_train()
with_evolution_test_loss = with_evolution_model.get_metric_on_test(test_features, test_colors)

plt.bar(['base', 'with graph', 'with evolution'], [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
plt.title('MSE on train set')
plt.show()
plt.bar(['base', 'with graph', 'with evolution'], [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
plt.title('MSE on test set')
plt.show()

evolution.plot_evolution_fitnesses()
evolution.base_individ.show_2d(labels=train_colors, euclidean=True)
evolution.base_individ.show_2d(labels=train_colors, euclidean=False)
