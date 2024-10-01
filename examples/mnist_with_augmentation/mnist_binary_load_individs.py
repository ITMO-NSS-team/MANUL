import os
from mnist_binary_n_runs import get_data, split_dataset

from evolution.IndividStructures import DataStructureGraph
from evolution.PopulationEvoOperators import Population


if __name__ == "__main__":
    feature, target, angle = get_data()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)
    train_angle, test_angle = split_dataset(angle)

    f_folder = "../../cache_mnist_n_runs/_30_10_mnist_2class" # path to directory with results

    for directory in os.listdir(f_folder):
        path_to_file = f"{f_folder}/{directory}/best_individs_by_iterations.pkl" # name of file with best individs
        if not os.path.isfile(path_to_file): continue
        instance_graph = DataStructureGraph(data=train_features, 
                                            cache_folder=f"{f_folder}/{directory}",
                                            graph_file='base_graph.pkl')

        pop_inidivids = Population(size=1, base_individ=instance_graph)
        pop_inidivids.load_individs_pool(path_to_file)

        pop_inidivids.individs_pool[-1].show_2d(labels=train_target, euclidean=True)