import ast
import numpy as np
import time

import torch
from progress.bar import Bar
from numba.typed import Dict
import numba.types as tp
from numba import njit, float64, int64, int32
from datetime import datetime
from copy import deepcopy
from functools import singledispatchmethod

import topo as tp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve
import plotly.graph_objects as go
import torch.nn as nn
from torch import randperm, tensor, mean, sqrt
from torch.optim import Adam
from torch import float64 as fl64
from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error
from scipy.optimize import minimize
from sklearn.decomposition import PCA

from evolution.IndividStructures import DataStructureGraph


class ModelNN:
    def __init__(self, train_feature: np.ndarray,
                 train_target: np.ndarray,
                 num_epochs: int,
                 problem: str,
                 batch_size: int = 300,
                 stop_criteria_count: int = 10,
                 criterion=None,
                 optimizer=None):
        self.device = self.init_device()
        self.features = train_feature
        self.target = train_target
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.problem = problem

        self.model = self._init_baseline_model(problem).to(self.device)
        self.optimizer, self.criterion = self._init_training_settings(criterion, optimizer)

        self.threshold = None  # parameter for classification problem
        self.stop_criteria_count = stop_criteria_count

    def init_device(self, device: str = None):
        """
        :param device: str - name of device
        """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return device

    def _init_baseline_model(self, problem):
        """
        Function for initialization network model structure based on problem
        """
        dim = self.features.shape[-1]
        sequence = [nn.Linear(dim, 512, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(512, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 256, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(256, 64, dtype=fl64),
                    nn.ReLU(),
                    nn.Linear(64, 1, dtype=fl64)]
        if problem == 'class':
            sequence.append(nn.Sigmoid())
        model = nn.Sequential(*sequence)
        return model

    def _init_training_settings(self, criterion, optimizer):
        """
        Function for setting optimizer and criterion for optimization network based on problem
        """
        if self.problem == 'class' and criterion is None:
            criterion = nn.BCELoss()
        if self.problem == 'regres' and criterion is None:
            criterion = nn.L1Loss()
        if optimizer is None:
            optimizer = Adam(self.model.parameters(), lr=1e-4, eps=1e-4)
        return optimizer, criterion

    def _calc_threshold_classification_problem(self, target, predicted):
        fpr, tpr, thresholds = roc_curve(target.reshape(-1), predicted.detach().numpy().reshape(-1))
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        if not self.threshold:
            self.threshold = thresholds[ix]
        else:
            self.threshold = np.mean([thresholds[ix], self.threshold])

    def _check_stop_criteria_on_graph(self, last_loss: float, current_loss: float, no_changes_counter, tolerance=0.01):
        """
        Function to check if loss function changes are significant
        """
        if last_loss is None:
            last_loss = current_loss
        else:
            if abs(current_loss - last_loss) <= tolerance:
                print(f'Stop criteria {no_changes_counter} / {self.stop_criteria_count}')
                no_changes_counter += 1
            last_loss = current_loss
        return last_loss, no_changes_counter

    def train(self, graph: DataStructureGraph = None):
        """
        :param graph: graph for additional loss calculation
        """
        if graph is not None:
            lmd = 1 / (self.batch_size ** 2)  # lambda as weight coefficient for custom graph loss

        self.model.train()

        epoch = 0
        last_loss = None
        no_changes_epoch = 0
        best_model = None
        best_loss = np.inf
        while epoch < self.num_epochs and no_changes_epoch <= self.stop_criteria_count:
            permutation = randperm(self.features.shape[0])
            loss_list = []
            for i in range(0, len(self.target), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, target_y = self.features[indices], self.target[indices]
                batch_x = torch.Tensor(batch_x).to(fl64).to(self.device)
                target_y = torch.Tensor(target_y).to(fl64).to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, target_y.reshape_as(output))

                if graph is not None:
                    add_loss = graph.loss_function(output.cpu().detach().numpy(), indices)
                    loss += lmd * tensor(add_loss)
                if self.problem == 'class':
                    self._calc_threshold_classification_problem(target_y, output)

                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            loss_epoch_mean = np.mean(loss_list)
            if loss_epoch_mean < best_loss:
                best_model = self.model
                best_loss = loss_epoch_mean
                print('Upd best model')
            print(f'Epoch: {epoch} / Loss = {np.round(loss_epoch_mean, 5)}')
            if graph is not None:
                last_loss, no_changes_epoch = self._check_stop_criteria_on_graph(last_loss, loss_epoch_mean, no_changes_epoch)

            epoch += 1
        self.model = best_model
        self.model.eval()

    def get_loss_on_train(self):
        output = self.model(torch.tensor(self.features).to(self.device)).cpu().detach().numpy()[:, 0]
        target_y = self.target.astype(float)
        if self.problem == 'class':
            return_loss = roc_auc_score(target_y, output)
            return return_loss
        return_loss = mean_squared_error(target_y, output)
        return return_loss
