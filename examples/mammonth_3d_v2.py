import ast

import numpy as np
import torch


from evolution.Evolution import Evolution
from evolution.PopulationStructures import Population
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def form_dataset():
    fl = open("data/mammoth_3d.json ", "r")
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


def find_graph_loss_raw(graph_laplassian, f_x, indexs=None):
    # кастомная функция для расчета фитнесса для графа
    if indexs is None:
        laplassian = graph_laplassian
    else:
        laplassian = graph_laplassian[indexs][:, indexs]
    part_1 = np.dot(f_x.T, laplassian)
    loss = np.dot(part_1, f_x)
    return loss.reshape(-1)[0]



def run_example():
    feature, target = form_dataset()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)
    '''train_features = torch.from_numpy(train_features)
    test_features = torch.from_numpy(test_features)
    train_target = torch.from_numpy(train_target)
    test_target = torch.from_numpy(test_target)'''

    base_individ = DataStructureGraph(data=train_features,
                                      n_neighbors=10,
                                      eps=0.15,
                                      cash_folder='C:/Users/Julia/Documents/NSS_lab/fastnet/examples/info_log/2024_04_21-04_10_19_PM',
                                      graph_file='base_graph')

    # считаем для простой нейронки без графа
    base_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                         num_epochs=50,
                         batch_size=300, problem='regres')
    base_model.train()
    base_train_loss = base_model.get_loss_on_train()

    # считаем для простой нейронки с базовым графом
    with_graph_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                         num_epochs=50,
                         batch_size=300, problem='regres')
    with_graph_model.train(base_individ)
    with_graph_train_loss = with_graph_model.get_loss_on_train()

    # считаем для кучи нейронок для каждого индивида в популяции с выбором лучшей модели

    with_evolution_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                         num_epochs=50,
                         batch_size=300, problem='regres')

    evolution = Evolution(base_individ=base_individ,
                          iterations=50,
                          population_size=10,
                          model_to_optimize=with_evolution_model)
    evolution.run()









run_example()
