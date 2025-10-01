import torch.nn as nn
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from torch import float64 as fl64


def nn_presets(problem: str, input_dim: int):
    """
    Function to return Sequential model based on task
    :param problem: string with problem name
    :param input_dim: input features dimensions num
    :return: model as nn.Sequential
    """
    presets_dict = {'regres': [nn.Linear(input_dim, 512, dtype=fl64),
                               nn.ReLU(),
                               nn.Linear(512, 256, dtype=fl64),
                               nn.ReLU(),
                               nn.Linear(256, 256, dtype=fl64),
                               nn.ReLU(),
                               nn.Linear(256, 64, dtype=fl64),
                               nn.ReLU(),
                               nn.Linear(64, 1, dtype=fl64)],
                    'binary_class': [nn.Linear(input_dim, 512, dtype=fl64),
                                     nn.ReLU(),
                                     nn.Linear(512, 256, dtype=fl64),
                                     nn.ReLU(),
                                     nn.Linear(256, 256, dtype=fl64),
                                     nn.ReLU(),
                                     nn.Linear(256, 64, dtype=fl64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1, dtype=fl64),
                                     nn.Sigmoid()],
                    'multiclass': [nn.Linear(input_dim, 512, dtype=fl64),
                                   nn.ReLU(),
                                   nn.Linear(512, 128, dtype=fl64),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.25),
                                   nn.Linear(128, 10, dtype=fl64),
                                   nn.Softmax(dim=1)]}
    seq = presets_dict[problem]
    model = nn.Sequential(*seq)
    return model


def criterion_presets(problem: str):
    presets_dict = {'regres': nn.L1Loss,
                    'binary_class': nn.BCELoss,
                    'multiclass': nn.CrossEntropyLoss}
    criterion = presets_dict[problem]()
    return criterion


def metrics_presets(problem: str):
    presets_dict = {'regres': {'def': mean_squared_error, 'params': {}},
                    'binary_class': {'def': roc_auc_score, 'params': {}},
                    'multiclass': {'def': f1_score, 'params': {'average': 'weighted'}}}
    metric = presets_dict[problem]
    return metric
