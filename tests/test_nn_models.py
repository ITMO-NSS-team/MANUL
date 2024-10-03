import os.path
import numpy as np
import pandas as pd
from torch import nn
import pickle as pkl

from tests.utils import create_model_circle_withoutgraph, create_model_circle_withgraph
from tests.utils import simple_nn, split_dataset, fake_loss
from regularizator.ModuleNN import ModelNN
from evolution.Evolution import Evolution


def get_synthetic_data(type: str):
    np.random.seed(1)  # fixed for class balance, do not change
    data = pd.DataFrame()
    data['feature1'] = np.arange(0, 2, 0.1)
    data['feature2'] = data['feature1'] * 2
    if type == 'regression':
        data['feature3'] = data['feature1'] ** 2
        data['target'] = data['feature1'] + data['feature2'] + data['feature3']
    if type == 'classification_binary':
        data['feature3'] = np.random.randint(2, size=20)
        data['target'] = 1 - data['feature3']
    if type == 'classification_multiclass':
        target = np.random.randint(3, size=20)
        data['feature3'] = target * data['feature1'] - data['feature2']
        data['target'] = target
    data = data.to_numpy()
    return data


def test_regression_preset_run():
    data = get_synthetic_data('regression')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model = ModelNN(problem='regres',
                    num_epochs=200,
                    train_feature=train_features,
                    train_target=train_target)
    model.train(num_epochs=50)
    assert model.num_epochs == 50
    metric_train = model.get_metric_on_train()
    metric_test = model.get_metric_on_test(train_features, train_target)
    assert metric_test, metric_train

    prediction = model.predict(test_features)
    uniq_values = np.unique(prediction)
    assert uniq_values.shape[0] > 3


def test_binary_classification_preset_run():
    data = get_synthetic_data('classification_binary')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model = ModelNN(problem='binary_class',
                    num_epochs=200,
                    train_feature=train_features,
                    train_target=train_target)
    model.train(num_epochs=50)
    assert model.num_epochs == 50
    metric_train = model.get_metric_on_train()
    metric_test = model.get_metric_on_test(train_features, train_target)
    assert metric_test, metric_train

    prediction = model.predict(test_features)
    uniq_values = np.unique(prediction)
    assert uniq_values.shape[0] == 2


def test_multiclass_classification_preset_run():
    data = get_synthetic_data('classification_multiclass')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model = ModelNN(problem='multiclass',
                    num_epochs=200,
                    train_feature=train_features,
                    train_target=train_target)
    model.train(num_epochs=50)
    assert model.num_epochs == 50
    metric_train = model.get_metric_on_train()
    metric_test = model.get_metric_on_test(train_features, train_target)
    assert metric_test, metric_train

    prediction = model.predict(test_features)
    uniq_values = np.unique(prediction)
    assert uniq_values.shape[0] == 3


def test_custom_model_run():
    data = get_synthetic_data('regression')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model_structure = simple_nn(train_features.shape[1])

    model = ModelNN(model_structure=model_structure,
                    train_feature=train_features,
                    train_target=train_target,
                    criterion=nn.L1Loss(),
                    target_metric=fake_loss
                    )
    model.train(num_epochs=50)
    # custom model applied check
    assert model.problem is None
    # custom metric applying check
    assert model.get_metric_on_train() == 9999
    assert model.get_metric_on_test(test_features, test_target) == 9999
    # check weights save
    model.save_weights(path='test_model_weights.pt')
    assert os.path.exists('test_model_weights.pt')
    prediction = model.predict(test_features)
    assert prediction is not None

    # check is model weights loaded correctly
    new_model_structure = simple_nn(train_features.shape[1])
    new_model_object = ModelNN(model_structure=new_model_structure,
                               model_weights='test_model_weights.pt',
                               train_feature=train_features,
                               train_target=train_target,
                               criterion=nn.L1Loss(),
                               target_metric=fake_loss
                               )
    new_prediction = new_model_object.predict(test_features)
    os.remove('test_model_weights.pt')
    assert (new_prediction == prediction).all()


def test_cache_folder():
    data = get_synthetic_data('regression')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model_structure = simple_nn(train_features.shape[1])

    model = ModelNN(model_structure=model_structure,
                    train_feature=train_features,
                    train_target=train_target,
                    criterion=nn.L1Loss(),
                    target_metric=fake_loss,
                    cache_folder='test_cache',
                    model_name='test_name'
                    )
    model.train(plot_convergence=True)
    model.save_weights()
    assert os.path.exists('test_cache/test_name_conv_plot.png')
    assert os.path.exists('test_cache/test_name.pt')
    os.remove('test_cache/test_name_conv_plot.png')
    os.remove('test_cache/test_name.pt')
    os.removedirs('test_cache')


def test_model_with_graph():
    model, individ_shell, train_features = create_model_circle_withgraph()

    model.train(graph=individ_shell, adaptive_lambda=False)
    assert np.all(model.features == train_features)
    assert len(model.trained_loss_values.keys()) == 3
    assert np.isclose(model.trained_loss_values['graph_loss'] + model.trained_loss_values['model_loss'], model.trained_loss_values['combined_loss'], atol=1e-4)

    

def test_model_with_evol():
    model, individ_shell, train_features = create_model_circle_withgraph()
    
    evolution = Evolution(base_individ=individ_shell,
                              iterations=1,
                              population_size=10,
                              model_to_optimize=model,
                              edges_weight_mutation=True)
    
    assert id(evolution.base_model) == id(model)
    evolution.run()
    assert np.all(model.features == train_features) and  np.all(model.features == evolution.base_individ.source_data)


def test_check_stop_criteria():
    model = create_model_circle_withoutgraph()

    last_loss = 1.2235256
    current_loss = 1.2235236

    last_loss, change_count = model._check_stop_criteria(last_loss, current_loss, 1)

    assert change_count == 2
    assert last_loss == current_loss

    current_loss = 5.3121

    last_loss, change_count = model._check_stop_criteria(last_loss, current_loss, change_count)

    assert change_count == 2
    assert last_loss == current_loss


def test_get_scaled_loss():
    model = create_model_circle_withoutgraph()

    loss_list = np.array([1,2,3,5,4])

    scale_loss = model._get_scaled_loss(loss_list)

    assert np.all(loss_list == np.array([1,2,3,5,4]))
    assert scale_loss == 0.75


def test_get_adaptive_lambda():
    model = create_model_circle_withoutgraph()

    combines_loss = [16.92036, 13.66173, 14.76129, 20.05487, 20.0401, 16.87396, 14.86376, 14.76943, 21.83021, 23.36205, 22.07374]
    graph_loss = [16.61611, 13.38675, 14.49282, 19.78522, 19.77816, 16.6297, 14.62064, 14.53423, 21.6005, 23.14165, 21.85664]
    nn_loss  = [0.30425, 0.27498, 0.26847, 0.26965, 0.26194, 0.24426, 0.24312, 0.23521, 0.22971, 0.22041, 0.2171]

    lmds = model._get_adaptive_lambda(combines_loss, nn_loss, graph_loss)
    check_lmds = np.array([0.43899881007134345, 1.0])

    assert np.all(np.isclose(lmds, check_lmds, atol=1e-4))