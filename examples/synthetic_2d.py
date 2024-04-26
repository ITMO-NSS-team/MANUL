import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def generate_ds():
    #  регулярная
    x = np.arange(0, 10000)
    y = np.arange(0, 10000)
    xs, ys = np.meshgrid(x, y)
    colors_image = np.sqrt(xs ** 2 + ys ** 2)

    # обучающая
    np.random.seed(0)
    x = np.random.randint(10000, size=3000)
    y = np.random.randint(10000, size=3000)
    train_colors = np.sqrt(x ** 2 + y ** 2)
    train_features = [np.ravel(x), np.ravel(y)]

    plt.imshow(colors_image)
    plt.scatter(train_features[0], train_features[1])
    plt.colorbar()
    plt.title('Train points')
    plt.show()

    # тестовая
    np.random.seed(1)
    x = np.random.randint(10000, size=100, )
    y = np.random.randint(10000, size=100)
    test_colors = np.sqrt(x ** 2 + y ** 2)
    test_features = [np.ravel(x), np.ravel(y)]

    plt.imshow(colors_image)
    plt.scatter(test_features[0], test_features[1])
    plt.colorbar()
    plt.title('Test points')
    plt.show()
    return colors_image, np.array(train_features).T.astype(float), np.array(train_colors).astype(float), np.array(test_features).T.astype(float), np.array(test_colors).astype(float)

colors_image, train_features, train_colors, test_features, test_colors = generate_ds()

base_individ = DataStructureGraph(data=train_features,
                                  n_neighbors=10,
                                  eps=0.3,
                                  cash_folder='C:/Users/Julia/Documents/NSS_lab/fastnet/examples/info_log/2d_sinthetic',
                                  graph_file='base_graph.pkl')

positions = base_individ.source_data[base_individ.basis]

g = nx.Graph()
for n in range(base_individ.number_of_nodes):
    g.add_node(n)
for n in base_individ.graph.keys():
    for e in base_individ.graph[n]:
        g.add_edge(n, e)

labels = {}
for k in range(base_individ.number_of_nodes):
    labels[k] = base_individ.basis[k]

plt.imshow(colors_image)
target = train_colors
plt.colorbar()
plt.scatter(train_features[:, 0], train_features[:, 1], c='blue')
plt.title('Base graph')
nx.draw(g, pos=positions, labels=labels, node_color='green')
plt.show()

base_model = ModelNN(train_features[base_individ.basis], target[base_individ.basis],
                     num_epochs=50,
                     batch_size=10, problem='regres')
base_model.train(plot_convergence=True)
base_train_loss = base_model.get_loss_on_train()
base_test_loss = base_model.get_loss_on_test(test_features, test_colors)

with_graph_model = ModelNN(train_features[base_individ.basis], target[base_individ.basis],
                           num_epochs=50,
                           batch_size=10, problem='regres')
with_graph_model.train(base_individ, plot_convergence=True)
with_graph_train_loss = with_graph_model.get_loss_on_train()
with_graph_test_loss = with_graph_model.get_loss_on_test(test_features, test_colors)

with_evolution_model = ModelNN(train_features[base_individ.basis], target[base_individ.basis],
                               num_epochs=50,
                               batch_size=10, problem='regres')

operators_params = {
                          'elitism': {'elits_num': 1},
                          'roulette_wheel_selection': {'tournament_size': None, 'winners_size': None},
                          'crossover': {'crossover_size_percent': 0.5},
                          'mutation': {'mutation_prob': 0.5}
                              }

evolution = Evolution(base_individ=base_individ,
                      iterations=10,
                      population_size=10,
                      model_to_optimize=with_evolution_model,
                      evo_operators_params=operators_params
                      )
evolution.run()

with_evolution_train_loss = with_evolution_model.get_loss_on_train()
with_evolution_test_loss = with_evolution_model.get_loss_on_test(test_features, test_colors)

plt.bar(['base', 'with graph', 'with evolution'], [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
plt.title('MSE on train set')
plt.show()
plt.bar(['base', 'with graph', 'with evolution'], [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
plt.title('MSE on test set')
plt.show()


evolution.plot_evolution_fitnesses()
ev_hist = evolution.evolution_history

# отрисовка лучшего графа
graph = evolution.base_individ
positions = base_individ.source_data[graph.basis]
g = nx.Graph()
for n in list(graph.graph.keys()):
    g.add_node(n)
for n in list(graph.graph.keys()):
    for e in graph.graph[n]:
        g.add_edge(n, e)
labels = {}
for k in range(g.number_of_nodes()):
    labels[k] = graph.basis[k]

plt.imshow(colors_image)
plt.colorbar()
plt.scatter(train_features[:, 0], train_features[:, 1], c='blue')
plt.title(f'Final topology\nFitness = {graph.fitness}')
nx.draw(g, pos=positions, labels=labels, node_color='green')
plt.show()

# специфичное для двумерных данных
for generation_num in ev_hist.keys():
    for graph_num in list(ev_hist[generation_num].keys()):
        graph = ev_hist[generation_num][graph_num]
        positions = base_individ.source_data[graph['basis']]
        g = nx.Graph()
        for n in list(graph['graph'].keys()):
            g.add_node(n)
        for n in list(graph['graph'].keys()):
            for e in graph['graph'][n]:
                g.add_edge(n, e)
        labels = {}
        for k in range(g.number_of_nodes()):
            labels[k] = graph['basis'][k]

        plt.imshow(colors_image)
        plt.colorbar()
        plt.scatter(train_features[:, 0], train_features[:, 1], c='blue')
        plt.title(f'Generation {generation_num}\nIndivid {graph_num}\nFitness = {graph["fitness"]}')
        nx.draw(g, pos=positions, labels=labels, node_color='green')
        plt.show()


