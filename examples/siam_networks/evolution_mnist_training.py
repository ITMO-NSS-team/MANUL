from torchvision import datasets
from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from structure_approximation.ModuleNN import ModelNN

cache_folder = 'cache'

dataset = datasets.MNIST('data', train=True, download=False)
train_data = dataset.train_data.numpy()
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = dataset.train_labels.numpy()

train_labels = train_labels[:30000]
train_data = train_data[:30000, :]

base_individ = DataStructureGraph(data=train_data,
                                  cache_folder=cache_folder,
                                  n_neighbors=20,
                                  epsilon_neighborhood=0.5,
                                  graph_file='base_graph.pkl')

# GRAPH MODEL
graph_model = ModelNN(train_data, train_labels,
                      num_epochs=200,
                      batch_size=300,
                      problem='multiclass',
                      cache_folder=cache_folder,
                      model_name='graph_model')
graph_model.train(base_individ)

with_evolution_model = ModelNN(train_data, train_labels,
                               num_epochs=200,
                               batch_size=300,
                               problem='multiclass',
                               cache_folder=cache_folder,
                               model_name='evo_graph_model')

evo_graph = DataStructureGraph(cache_folder=cache_folder, graph_file='final_graph.pkl')
with_evolution_model.train(evo_graph)

evolution = Evolution(base_individ=base_individ,
                      iterations=50,
                      population_size=10,
                      model_to_optimize=with_evolution_model,
                      edges_weight_mutation=True)
evolution.run()
evolution.base_individ.show_2d(train_labels, save_path=f'{cache_folder}/final_graph.png')
evolution.plot_evolution_fitnesses(save_path=f'{cache_folder}/evolution_conv.png')
