import numpy as np
from matplotlib import pyplot as plt
from evolution.Evolution import Evolution
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


def get_data():
    features = np.load("data/feature_mnist.npy")
    target = np.load("data/target_mnist.npy")
    angles = np.load("data/angle_mnist.npy")
    # data is already shuffled for class balance
    new_features = features.reshape((features.shape[0], features.shape[1] * features.shape[2]))
    new_feature = []
    new_target = []
    new_angles = []
    for i, elem in enumerate(target):
        if elem not in [3, 4, 5, 6, 7, 8, 9]:
            # remove 9 from classification because augmentation makes it equal to 6
            new_feature.append(new_features[i])
            new_target.append(elem)
            new_angles.append(angles[i])
    samples_num = 20000
    new_feature = np.array(new_feature[:samples_num], dtype='int64')
    #new_feature[new_feature != 0] = 1
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


def run_example():
    feature, target, angles = get_data()
    train_features, test_features = split_dataset(feature)
    train_target, test_target = split_dataset(target)

    train_angles, test_angles = split_dataset(angles)
    train_angles = form_markers_by_angle(train_angles)
    test_angles = form_markers_by_angle(test_angles)


    base_individ = DataStructureGraph(data=train_features,
                                      cash_folder='C:/Users/Julia/Documents/NSS_lab/fastnet/examples/info_log/mnist_3class_fix',
                                      n_neighbors=20
                                      )
    base_individ.show_3d(labels=train_target, markers=train_angles, title='Before evolution')
    base_individ.show_2d(labels=train_target, euclidean=True)

    # считаем для простой нейронки без графа
    base_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                         num_epochs=50,
                         batch_size=300, problem='multiclass')
    base_model.train()
    base_train_loss = base_model.get_loss_on_train()
    base_test_loss = base_model.get_loss_on_test(test_features, test_target)

    # считаем для простой нейронки с базовым графом
    with_graph_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                               num_epochs=50,
                               batch_size=300, problem='multiclass')
    with_graph_model.train(base_individ)
    with_graph_train_loss = with_graph_model.get_loss_on_train()
    with_graph_test_loss = base_model.get_loss_on_test(test_features, test_target)

    # считаем для кучи нейронок для каждого индивида в популяции с выбором лучшей модели
    with_evolution_model = ModelNN(train_features[base_individ.basis], train_target[base_individ.basis],
                                   num_epochs=50,
                                   batch_size=300, problem='multiclass')

    evolution = Evolution(base_individ=base_individ,
                          iterations=30,
                          population_size=7,
                          model_to_optimize=with_evolution_model)
    evolution.run()
    evolution.plot_evolution_fitnesses()

    evolution.base_individ.show_2d(train_target, euclidean=True)
    evolution.base_individ.show_3d(train_target, markers=train_angles, title='After evolution')

    with_evolution_train_loss = with_evolution_model.get_loss_on_train()
    with_evolution_test_loss = with_evolution_model.get_loss_on_test(test_features, test_target)

    plt.bar(['base', 'with graph', 'with evolution'],
            [base_train_loss, with_graph_train_loss, with_evolution_train_loss])
    plt.title('Accuracy on train set')
    plt.show()
    plt.bar(['base', 'with graph', 'with evolution'], [base_test_loss, with_graph_test_loss, with_evolution_test_loss])
    plt.title('Accuracy on test set')
    plt.show()


run_example()