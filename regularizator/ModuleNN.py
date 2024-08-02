import os
from datetime import datetime
from typing import Callable
from tqdm import tqdm

import numpy as np
from SALib import ProblemSpec
import torch
from matplotlib import pyplot as plt

import torch.nn as nn
from torch import randperm, tensor
from torch.optim import Adam
from torch import float64 as fl64

from evolution.IndividStructures import DataStructureGraph

import warnings

from regularizator.models_presets import nn_presets, criterion_presets, metrics_presets

warnings.filterwarnings('ignore', module='SALib')  # pandas deprecation warning
warnings.filterwarnings('ignore', module='numpy')  # empty slice mean


class ModelNN:
    def __init__(self, train_feature: np.ndarray,
                 train_target: np.ndarray,
                 problem: str = None,
                 model_structure: nn.Sequential = None,
                 model_weights: str = None,
                 num_epochs: int = 100,
                 batch_size: int = 300,
                 stop_criteria_count: int = 10,
                 criterion=None,
                 optimizer=None,
                 target_metric: Callable = None,
                 cash_folder: str = None,
                 model_name: str = None):
        """
        :param train_feature: array with features for model training
        :param train_target: array with target for model training
        :param num_epochs: number of epochs for model training
        :param problem: "regres","binary_class", "multiclass" are available
        :param batch_size: batch size for model training
        :param stop_criteria_count: number of low loss changes epochs for training stop
        :param criterion: torch loss function (for custom training)
        :param optimizer: torch optimizer (for custom training)
        :param target_metric: function to calculate metric on prediction and target
                                          (default regres - mse, binary class - roc_auc, multiclass - accuracy)
        :param cash_folder: string with cash folder to save convergence plots (if empty plots doesn't save)
        :param model_name: string with model name to save on plot
        """
        self.model_name = model_name
        if self.model_name is None:
            self.model_name = f"{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}_model"

        self.trained_loss_values = {'model_loss': None,
                                    'graph_loss': None,
                                    'combined_loss': None}

        self.device = self.init_device()
        self.features = train_feature.astype(float)
        self.target = train_target
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.problem = problem

        self.init_model(model_structure, model_weights)

        self._init_training_settings(criterion, optimizer)
        self._init_target_metric(target_metric)

        self.stop_criteria_count = stop_criteria_count
        if cash_folder is not None:
            if not os.path.exists(cash_folder):
                os.mkdir(cash_folder)
        self.cash_folder = cash_folder
        self.model_name = model_name

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

    def _init_target_metric(self, target_metric: [Callable, None]):
        self.target_metric = target_metric
        if self.target_metric is None:
            if self.problem is None:
                raise Exception('Use callable as metric function in "target_metric" or specify "problem" to use model '
                                'preset')
            else:
                self.target_metric = metrics_presets(self.problem)

    def init_model(self, model_structure: [nn.Sequential, None], model_weights: [str, None]):
        """
        Function to load custom model and its weights if exist
        :param model_structure: sequence with model structure
        :param model_weights: path to .pt file with model weights
        """
        if model_structure is not None:
            self.model = model_structure
            if model_weights is not None:
                self.model.load_state_dict(torch.load(model_weights))
                self.model.eval()
                print('Model weights loaded')
        else:
            if self.problem is None:
                raise Exception('Use custom model structure in "model_structure" or specify "problem" to use model '
                                'preset')
            else:
                self._init_baseline_model()
        self.model = self.model.to(self.device)

    def _init_baseline_model(self):
        """
        Function for initialization network model structure based on problem
        """
        if self.problem not in ['regres', 'binary_class', 'multiclass']:
            raise Exception(f'No base model for problem - {self.problem} implemented, '
                            f'available problems: "regres", "binary_class", "multiclass" ')
        input_dim = self.features.shape[-1]
        self.model = nn_presets(self.problem, input_dim).to(self.device)

    def save_weights(self, path: str = None):
        """
        Function for saving model weights
        """
        if path is not None:
            torch.save(self.model.state_dict(), path)
        elif self.cash_folder is not None:
            torch.save(self.model.state_dict(), f'{self.cash_folder}/{self.model_name}.pt')
        else:
            torch.save(self.model.state_dict(), f'{self.model_name}.pt')

    def _init_training_settings(self, criterion, optimizer):
        """
        Function for setting optimizer and criterion for optimization network based on problem
        """
        self.criterion = criterion
        if self.criterion is None:
            if self.problem is None:
                raise Exception('Use criterion for NN model in "criterion" or specify "problem" to use model preset')
            else:
                self.criterion = criterion_presets(self.problem)
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = Adam(self.model.parameters(), lr=1e-3, eps=1e-4)

    def _check_stop_criteria(self, last_loss: float,
                             current_loss: float,
                             no_changes_counter: int,
                             tolerance: float = 0.0001):
        """
        Function to check if loss function changes are significant
        """
        if last_loss is None:
            last_loss = current_loss
        else:
            if abs(current_loss - last_loss) <= tolerance:
                no_changes_counter += 1
            last_loss = current_loss
        return last_loss, no_changes_counter

    def _get_scaled_loss(self, loss_list):
        """
        Function for dynamical scaling loss on previous values of loss
        :param loss_list: list with raw values to scale
        :return scaled last loss value
        """
        if loss_list.shape[0] == 1:
            return 1
        loss_scaled = (loss_list - loss_list.min()) / (loss_list.max() - loss_list.min())
        return loss_scaled[-1]

    def _get_adaptive_lambda(self, combines_loss, nn_loss, graph_loss):
        """
        :param combines_loss:  matrix m x n where m - epochs number, n - batch size with sum of nn and graph losses
        :param nn_loss: matrix m x n where m - epochs number, n - batch size with graph losses
        :param graph_loss: matrix m x n where m - epochs number, n - batch size with nn losses
        :return: list [float, float] - list with coefficients to multiply with nn loss and graph loss
        """
        n_samples = 1  # can be changed to use more elements of lists
        sampling_D = 2  # as combine 2 features

        if n_samples * (sampling_D * 2 + 2) > len(combines_loss):
            print('Epochs number is too small to calculate adaptive lambda')
            return [1, 1]

        combines_loss = np.array(combines_loss)
        nn_loss = np.expand_dims(np.array(nn_loss), axis=1)
        graph_loss = np.expand_dims(np.array(graph_loss), axis=1)

        X_array = np.hstack((nn_loss, graph_loss))

        bounds = [[-100, 100] for i in range(sampling_D)]
        names = ['x{}'.format(i) for i in range(sampling_D)]

        X_array = X_array[:n_samples * (X_array.shape[1] * 2 + 2)]
        combines_loss = combines_loss[:n_samples * (X_array.shape[1] * 2 + 2)]

        sp = ProblemSpec({'names': names, 'bounds': bounds})
        sp.set_samples(X_array)
        sp.set_results(combines_loss)
        sp.analyze_sobol(calc_second_order=True)

        ST = sp.analysis['ST']
        total_disp = sum(ST)

        nn_disp = sum(ST[:nn_loss.shape[1]])
        graph_disp = sum(ST[nn_loss.shape[1]:])

        if nn_disp == 0 or graph_disp == 0:
            print(f'Lambda search failed: nn_disp={nn_disp}, graph_disp={graph_disp}')
            return [1, 1]

        lam_nn = total_disp / nn_disp
        lam_graph = total_disp / graph_disp

        if np.isnan(lam_nn) or np.isnan(lam_graph):
            print(f'Lambda search failed: nn_disp={lam_nn}, graph_disp={lam_graph}')
            return [1, 1]
        
        return [lam_nn / (np.nanmax([lam_nn, lam_graph])), lam_graph / (np.nanmax([lam_nn, lam_graph]))]

    def preprocess_target(self, nn_output, target_y: np.ndarray):
        """
        Function to reshape and preprocess target to output format based on task
        """
        if self.problem == 'multiclass':
            temp = np.zeros(nn_output.shape)
            trans_target = target_y.astype('int')
            temp[np.arange(target_y.shape[0]).astype(int), trans_target] = 1
            target_y = torch.Tensor(temp).to(fl64).to(self.device)
        else:
            target_y = torch.Tensor(target_y).to(fl64).to(self.device)
        return target_y

    def train(self, graph: DataStructureGraph = None,
              plot_convergence=False,
              lmds: list[float, float] = None,
              weight_loss: bool = False,
              adaptive_lambda: bool = True,
              num_epochs: int = None):
        """
        :param num_epochs: number of epochs for model training
        :param adaptive_lambda: flag to calculate adaptive weights for combined loss on part of epochs
        :param weight_loss: flag to use dynamical weighting (by scaling) of two parts of combined loss
        :param lmds: lambdas value - weight coefficients for combined loss - [nn lmd, graph lmd]
        :param graph: graph for additional loss calculation
        :param plot_convergence: flag for plotting of mean epoch loss value
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs

        if lmds is None:
            lmds = [1, 1]

        if graph is not None:
            self.features = self.features[graph.basis]
            self.target = self.target[graph.basis]

        self.model.train()

        epoch = 0
        lmds_epochs = None
        last_loss = None
        no_changes_epoch = 0
        losses = []
        graph_losses = []
        nn_losses = []

        progress_bar = tqdm(list(np.arange(self.num_epochs)), desc="Epoch", colour="white")
        info_bar = {"Loss": 0}

        while epoch < self.num_epochs and no_changes_epoch <= self.stop_criteria_count:
            permutation = randperm(self.features.shape[0])
            loss_list = np.array([])
            graph_loss_list = np.array([])
            nn_loss_list = np.array([])

            for i in range(0, len(self.target), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_x, target_y = self.features[indices], self.target[indices]
                batch_x = torch.Tensor(batch_x).to(fl64).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)
                target_y = self.preprocess_target(output, target_y)
                loss = self.criterion(output, target_y.reshape_as(output))
                nn_loss_list = np.append(nn_loss_list, loss.item())

                if graph is not None:
                    add_loss = graph.loss_function(output.cpu().detach().numpy(), indices)
                    graph_loss_list = np.append(graph_loss_list, add_loss)

                    if adaptive_lambda:
                        lmds_epochs = int(self.num_epochs * 0.1)
                        if epoch <= lmds_epochs:  # 10% of epochs used to find lambdas
                            loss = lmds[0] * loss + lmds[1] * tensor(add_loss)
                        if epoch > lmds_epochs:  # then lambdas are used as new constants
                            lmds = self._get_adaptive_lambda(losses, nn_losses, graph_losses)
                            info_bar['nn_lmd'] = np.round(lmds[0], 5)
                            info_bar['graph_lmd'] = np.round(lmds[1], 5)
                            adaptive_lambda = False

                    if weight_loss and not adaptive_lambda:
                        nn_loss = self._get_scaled_loss(nn_loss_list)
                        add_loss = self._get_scaled_loss(graph_loss_list)
                        loss = loss - (tensor(loss.item())) + tensor(nn_loss) + tensor(add_loss)
                    if not weight_loss and not adaptive_lambda:
                        loss = lmds[0] * loss + lmds[1] * tensor(add_loss)

                loss_list = np.append(loss_list, loss.item())

                loss.backward()
                self.optimizer.step()

            loss_epoch_mean = np.mean(loss_list)
            info_bar['Loss'] = np.round(loss_epoch_mean, 5)
            progress_bar.update()
            progress_bar.set_postfix_str(info_bar)

            losses.append(np.round(loss_epoch_mean, 5))
            graph_losses.append(np.round(np.mean(graph_loss_list), 5))
            nn_losses.append(np.round(np.mean(nn_loss_list), 5))

            last_loss, no_changes_epoch = self._check_stop_criteria(last_loss,
                                                                    loss_epoch_mean,
                                                                    no_changes_epoch)
            epoch += 1
        self.model.eval()
        self.trained_loss_values['combined_loss'] = losses[-1]
        if graph is not None:
            self.trained_loss_values['graph_loss'] = graph_losses[-1]
            self.trained_loss_values['model_loss'] = nn_losses[-1]
        if plot_convergence:
            if graph is None:
                graph_losses = None
                nn_losses = None
            self._plot_convergence(losses, lmds_epochs, nn_losses, graph_losses)

    def _plot_convergence(self, losses, lmds_epoch, nn_losses=None, graph_losses=None):
        """
        Function to plot model convergence on losses lists
        :param losses: list of combined loss values
        :param nn_losses: list of NN outputs loss values
        :param graph_losses: list of graph loss values
        :param lmds_epoch: number of epochs to mark them as lambda search
        """
        if graph_losses is None or nn_losses is None:
            fig, axs = plt.subplots(1, 1, figsize=(5, 4))
            axs = [axs]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        # plot combines losses
        axs[0].plot(np.arange(len(losses)), losses)
        if lmds_epoch is not None:
            z = np.polyfit(np.arange(len(losses[lmds_epoch:])), losses[lmds_epoch:], 1)
            axs[0].axvline(lmds_epoch, c='green', linewidth=0.5)
        else:
            z = np.polyfit(np.arange(len(losses)), losses, 1)
        p = np.poly1d(z)
        axs[0].plot(np.arange(len(losses)), p(np.arange(len(losses))), "r--")
        axs[0].set_title(f'Combined loss = {np.round(self.trained_loss_values["combined_loss"], 5)}')

        # plot graph losses
        if graph_losses is not None:
            axs[1].plot(np.arange(len(graph_losses)), graph_losses)
            if lmds_epoch is not None:
                z = np.polyfit(np.arange(len(graph_losses[lmds_epoch:])), graph_losses[lmds_epoch:], 1)
                axs[1].axvline(lmds_epoch, c='green', linewidth=0.5)
            else:
                z = np.polyfit(np.arange(len(graph_losses)), graph_losses, 1)
            p = np.poly1d(z)
            axs[1].plot(np.arange(len(graph_losses)), p(np.arange(len(graph_losses))), "r--")
            axs[1].set_title(f'Graph loss = {np.round(self.trained_loss_values["graph_loss"], 5)}')

        # plot nn losses
        if nn_losses is not None:
            axs[2].plot(np.arange(len(nn_losses)), nn_losses)
            if lmds_epoch is not None:
                z = np.polyfit(np.arange(len(nn_losses[lmds_epoch:])), nn_losses[lmds_epoch:], 1)
                axs[2].axvline(lmds_epoch, c='green', linewidth=0.5)
            else:
                z = np.polyfit(np.arange(len(nn_losses)), nn_losses, 1)
            p = np.poly1d(z)
            axs[2].plot(np.arange(len(nn_losses)), p(np.arange(len(nn_losses))), "r--")
            axs[2].set_title(f'NN loss = {np.round(self.trained_loss_values["model_loss"], 5)}')

        for ax in axs:
            ax.set(xlabel='Epoch', ylabel='Loss value')

        fig.suptitle(f'Convergence plot')
        plt.tight_layout()
        if self.cash_folder is not None:
            plt.savefig(f'{self.cash_folder}/{self.model_name}_conv_plot.png')
            plt.close()
        plt.show()

    def _get_metric(self, true: np.ndarray, predicted: np.ndarray):
        """
        Function to calculate metric value for target and prediction
        """
        if isinstance(self.target_metric, dict):
            # if metric from preset
            metric_function = self.target_metric['def']
            params = self.target_metric['params']
            value = metric_function(true, predicted, **params)
        else:
            # if metric is custom callable
            value = self.target_metric(true, predicted)
        return value

    def get_metric_on_train(self):
        output = self.model(torch.tensor(self.features).to(self.device))
        target_y = self.target
        if self.problem == 'multiclass':
            output = output.cpu().detach().numpy()
            output = np.argmax(output, axis=1)
            target_y = self.target.astype(int)
        if self.problem == 'binary_class':
            output = output.cpu().detach().numpy()[:, 0]
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            target_y = self.target.astype(int)
        if self.problem == 'regres':
            output = output.cpu().detach().numpy()[:, 0]
            target_y = self.target.astype(float)
        return_loss = self._get_metric(target_y, output)
        return return_loss

    def get_metric_on_test(self, test_features, test_target):
        test_features = test_features.astype(float)
        target_y = test_target
        output = self.model(torch.tensor(test_features).to(self.device))
        if self.problem == 'multiclass':
            output = output.cpu().detach().numpy()
            output = np.argmax(output, axis=1)
            target_y = test_target.astype(int)
        if self.problem == 'binary_class':
            output = output.cpu().detach().numpy()[:, 0]
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            target_y = test_target.astype(int)
        if self.problem == 'regres':
            output = output.cpu().detach().numpy()[:, 0]
            target_y = test_target.astype(float)
        return_loss = self._get_metric(target_y, output)
        return return_loss

    def predict(self, test_features):
        output = self.model(torch.tensor(test_features).to(self.device)).cpu().detach().numpy()
        if self.problem == 'multiclass':
            output = np.argmax(output, axis=1)
        if self.problem == 'binary_class':
            output = output[:, 0]
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
        return output
