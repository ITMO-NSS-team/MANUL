import os.path
from torch import float64 as fl64
import numpy as np
import pandas as pd
from torch import nn

from regularizator.ModuleNN import ModelNN


def simple_nn(inp_dims):
    model = nn.Sequential(nn.Linear(inp_dims, 128, dtype=fl64),
                          nn.ReLU(),
                          nn.Linear(128, 1, dtype=fl64),
                          nn.ReLU())
    return model


def fake_loss(true, predicted):
    """
    Function to imitate the callable object of custom metric function
    """
    return 9999


def split_dataset(data, split_ratio=0.8):
    split_ratio = int(data.shape[0] * split_ratio)
    train = data[:split_ratio]
    test = data[split_ratio:]
    return train, test


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


def test_cash_folder():
    data = get_synthetic_data('regression')
    train_features, test_features = split_dataset(data[:, :-1])
    train_target, test_target = split_dataset(data[:, -1])

    model_structure = simple_nn(train_features.shape[1])

    model = ModelNN(model_structure=model_structure,
                    train_feature=train_features,
                    train_target=train_target,
                    criterion=nn.L1Loss(),
                    target_metric=fake_loss,
                    cash_folder='test_cash',
                    model_name='test_name'
                    )
    model.train(plot_convergence=True)
    model.save_weights()
    assert os.path.exists('test_cash/test_name_conv_plot.png')
    assert os.path.exists('test_cash/test_name.pt')
    os.remove('test_cash/test_name_conv_plot.png')
    os.remove('test_cash/test_name.pt')
    os.removedirs('test_cash')


def test_models_with_graph():
    # TODO create graph sample and run models
    pass

