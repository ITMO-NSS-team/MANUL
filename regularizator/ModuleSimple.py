from datetime import datetime
from typing import Callable
from tqdm import tqdm

import numpy as np
from SALib import ProblemSpec
import torch
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

import torch.nn as nn
from torch import randperm, tensor
from torch.optim import Adam
from torch import float64 as fl64
from sklearn.metrics import roc_auc_score, mean_squared_error

from evolution.IndividStructures import DataStructureGraph

import warnings

warnings.filterwarnings('ignore', module='SALib')  # pandas deprecation warning
warnings.filterwarnings('ignore', module='numpy')  # empty slice mean


class ModelSimple:
    def __init__(self, train_feature: np.ndarray,
                 train_target: np.ndarray,
                 problem: str,
                 target_metric: Callable = None,
                 cach_folder: str = None,
                 model_name: str = None):
        """
        :param train_feature: array with features for model training
        :param train_target: array with target for model training
        :param problem: "regres","binary_class", "multiclass" are available
        :param target_metric: function to calculate metric on prediction and target
                                          (default regres - mse, binary class - roc_auc, multiclass - accuracy)
        :param cach_folder: string with cach folder to save convergence plots (if empty plots doesn't save)
        :param model_name: string with model name to save on plot
        """
        self.trained_loss_values = {'model_loss': None,
                                    'graph_loss': None,
                                    'combined_loss': None}

        self.features = train_feature.astype(float)
        self.target = train_target
        self.problem = problem

        self.model = self._init_baseline_model(problem)
        self.target_metric = self._init_target_metric(target_metric)

        self.threshold = None  # parameter for classification problem
        self.cach_folder = cach_folder
        self.model_name = model_name

    def _init_target_metric(self, target_metric: [Callable, None]):
        if target_metric is None:
            if self.problem == 'regres':
                return 'mean_squared_error'
            if self.problem == 'binary_class':
                return 'roc_auc_score'
            if self.problem == 'multiclass':
                return 'f1_score'
        else:
            return target_metric

    def _init_baseline_model(self, problem):
        """
        Function for initialization network model structure based on problem
        """
        if problem not in ['regres', 'binary_class', 'multiclass']:
            raise Exception(f'No base model for problem - {problem} implemented, '
                            f'available problems: "regres", "binary_class", "multiclass" ')
        if problem == 'regres':
            model = LinearRegression()
        if problem == 'binary_class':
            model = SVC(kernel='linear', C=1.0)
        if problem == 'multiclass':
            model = LogisticRegression()
        return model

    def train(self, graph: DataStructureGraph = None,
              lmds: list[float, float] = None):
        if lmds is None:
            lmds = [1, 1]

        self.model = self.model.fit(self.features, self.target)
        output = self.model.predict(self.features)
        model_loss = self.get_metric_on_train()
        self.trained_loss_values['model_loss'] = model_loss
        if graph is not None:
            add_loss = graph.loss_function(output)
            loss = lmds[0] * model_loss + lmds[1] * add_loss
            self.trained_loss_values['graph_loss'] = add_loss
            self.trained_loss_values['combined_loss'] = loss

    def _get_metric(self, true: np.ndarray, predicted: np.ndarray):
        """
        Function to calculate metric value for target and prediction
        """
        if self.target_metric == 'mean_squared_error':
            return mean_squared_error(true, predicted)
        if self.target_metric == 'roc_auc_score':
            return roc_auc_score(true, predicted)
        if self.target_metric == 'f1_score':
            return f1_score(true, predicted, average='weighted')
        else:
            return self.target_metric(true, predicted)

    def get_metric_on_train(self):
        output = self.model.predict(self.features)
        if self.problem == 'multiclass' or self.problem == 'binary_class':
            target_y = self.target.astype(int)
        if self.problem == 'regres':
            target_y = self.target.astype(float)
        return_loss = self._get_metric(target_y, output)
        return return_loss

    def get_metric_on_test(self, test_features, test_target):
        output = self.model.predict(test_features)
        if self.problem == 'multiclass' or self.problem == 'binary_class':
            target_y = test_target.astype(int)
        if self.problem == 'regres':
            target_y = test_target.astype(float)
        return_loss = self._get_metric(target_y, output)
        return return_loss

    def predict(self, test_features):
        output = self.model.predict(test_features)
        return output
