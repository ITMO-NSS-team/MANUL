"""
Lambda Optimizers - методы адаптивной оптимизации коэффициента графовой регуляризации.

Реализованные методы:
1. FixedLambdaOptimizer - фиксированная lambda (baseline)
2. SobolLambdaOptimizer - анализ чувствительности Sobol (оригинальный метод)
3. GradNormOptimizer - балансировка через нормы градиентов (Chen et al., 2018)
4. UncertaintyWeightingOptimizer - через гомоскедастическую неопределённость (Kendall et al., 2018)
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

try:
    from SALib import ProblemSpec
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("Warning: SALib not available, SobolLambdaOptimizer will not work")


class BaseLambdaOptimizer(ABC):
    """
    Базовый класс для всех методов оптимизации lambda.

    """
    
    def __init__(self, initial_lambda_graph: float = 1.0):
        self.initial_lambda_graph = initial_lambda_graph
        self.lam_nn = 1.0
        self.lam_graph = initial_lambda_graph
        self._history: Dict[str, List[float]] = {
            'lam_nn': [],
            'lam_graph': [],
            'model_loss': [],
            'graph_loss': []
        }
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def get_lambdas(self) -> Tuple[float, float]:
        """
        Возвращает текущие веса (lam_nn, lam_graph).
        
        Returns:
            Tuple[float, float]: (вес для model_loss, вес для graph_loss)
        """
        pass
    
    @abstractmethod
    def update(self, 
               epoch: int,
               model_loss: float,
               graph_loss: float,
               model: Optional[nn.Module] = None,
               **kwargs) -> None:
        """
        Обновляет внутреннее состояние оптимизатора.
        
        Args:
            epoch: текущая эпоха
            model_loss: значение model loss за эпоху
            graph_loss: значение graph loss за эпоху
            model: модель (нужна для GradNorm)
            **kwargs: дополнительные параметры (градиенты и т.д.)
        """
        pass
    
    def reset(self) -> None:
        """Сброс состояния оптимизатора"""
        self.lam_nn = 1.0
        self.lam_graph = self.initial_lambda_graph
        self._history = {
            'lam_nn': [],
            'lam_graph': [],
            'model_loss': [],
            'graph_loss': []
        }
    
    def get_history(self) -> Dict[str, List[float]]:
        """Возвращает историю lambda и лоссов"""
        return self._history.copy()
    
    def _record_history(self, model_loss: float, graph_loss: float) -> None:
        """Записывает текущее состояние в историю"""
        self._history['lam_nn'].append(self.lam_nn)
        self._history['lam_graph'].append(self.lam_graph)
        self._history['model_loss'].append(model_loss)
        self._history['graph_loss'].append(graph_loss)


class FixedLambdaOptimizer(BaseLambdaOptimizer):
    """
    Фиксированная lambda - baseline метод.

    """
    
    def __init__(self, lambda_graph: float = 0.0):
        super().__init__(initial_lambda_graph=lambda_graph)
        self._name = f"baseline_lambda_{lambda_graph}"
    
    @property
    def name(self) -> str:
        return self._name
    
    def get_lambdas(self) -> Tuple[float, float]:
        return (self.lam_nn, self.lam_graph)
    
    def update(self, 
               epoch: int,
               model_loss: float,
               graph_loss: float,
               model: Optional[nn.Module] = None,
               **kwargs) -> None:

        self._record_history(model_loss, graph_loss)


class SobolLambdaOptimizer(BaseLambdaOptimizer):
    """
    Sobol Sensitivity Analysis
    """
    
    def __init__(self, 
                 initial_lambda_graph: float = 1.0,
                 warmup_fraction: float = 0.1,
                 n_samples: int = 5):
        super().__init__(initial_lambda_graph=initial_lambda_graph)
        self.warmup_fraction = warmup_fraction
        self.n_samples = n_samples
        self._warmup_done = False
        self._combined_losses: List[float] = []
        self._model_losses: List[float] = []
        self._graph_losses: List[float] = []
    
    @property
    def name(self) -> str:
        return "sobol"
    
    def get_lambdas(self) -> Tuple[float, float]:
        return (self.lam_nn, self.lam_graph)
    
    def update(self,
               epoch: int,
               model_loss: float,
               graph_loss: float,
               model: Optional[nn.Module] = None,
               num_epochs: int = 100,
               **kwargs) -> None:

        combined_loss = self.lam_nn * model_loss + self.lam_graph * graph_loss
        self._combined_losses.append(combined_loss)
        self._model_losses.append(model_loss)
        self._graph_losses.append(graph_loss)
        self._record_history(model_loss, graph_loss)

        warmup_epochs = int(num_epochs * self.warmup_fraction)
        
        if not self._warmup_done and epoch == warmup_epochs and SALIB_AVAILABLE:
            self._warmup_done = True
            new_lam_nn, new_lam_graph = self._compute_sobol_lambdas()
            if new_lam_nn is not None:
                self.lam_nn = new_lam_nn
                self.lam_graph = new_lam_graph
                print(f"  [Sobol] Updated lambdas at epoch {epoch}: "
                      f"lam_nn={self.lam_nn:.6f}, lam_graph={self.lam_graph:.6f}")
    
    def _compute_sobol_lambdas(self) -> Tuple[Optional[float], Optional[float]]:
        """Вычисляет оптимальные lambda через Sobol анализ"""
        
        n_samples = self.n_samples
        sampling_D = 2  # 2 фактора: model_loss и graph_loss
        
        min_required = n_samples * (sampling_D * 2 + 2)
        
        if len(self._combined_losses) < min_required:
            print(f"  [Sobol] Not enough data: {len(self._combined_losses)} < {min_required}")
            return None, None
        
        try:

            combined = np.array(self._combined_losses[:min_required])
            nn_loss = np.expand_dims(np.array(self._model_losses[:min_required]), axis=1)
            graph_loss = np.expand_dims(np.array(self._graph_losses[:min_required]), axis=1)
            
            X_array = np.hstack((nn_loss, graph_loss))
            
            bounds = [[-100, 100] for _ in range(sampling_D)]
            names = [f'x{i}' for i in range(sampling_D)]
            
            sp = ProblemSpec({'names': names, 'bounds': bounds})
            sp.set_samples(X_array)
            sp.set_results(combined)
            sp.analyze_sobol(calc_second_order=True)
            
            ST = sp.analysis['ST']
            total_disp = sum(ST)
            
            nn_disp = sum(ST[:nn_loss.shape[1]])
            graph_disp = sum(ST[nn_loss.shape[1]:])
            
            if nn_disp == 0 or graph_disp == 0:
                print(f"  [Sobol] Zero dispersion: nn_disp={nn_disp}, graph_disp={graph_disp}")
                return None, None
            
            lam_nn = total_disp / nn_disp
            lam_graph = total_disp / graph_disp
            
            if np.isnan(lam_nn) or np.isnan(lam_graph):
                print(f"  [Sobol] NaN values: lam_nn={lam_nn}, lam_graph={lam_graph}")
                return None, None

            max_lam = np.nanmax([lam_nn, lam_graph])
            lam_nn = lam_nn / max_lam
            lam_graph = lam_graph / max_lam
            
            return lam_nn, lam_graph
            
        except Exception as e:
            print(f"  [Sobol] Error: {e}")
            return None, None
    
    def reset(self) -> None:
        super().reset()
        self._warmup_done = False
        self._combined_losses = []
        self._model_losses = []
        self._graph_losses = []


class GradNormOptimizer(BaseLambdaOptimizer):
    """
    GradNorm - балансировка через нормы градиентов.
    
    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive 
               Loss Balancing in Deep Multitask Networks", ICML 2018
               https://arxiv.org/abs/1711.02257
    
    Args:
        alpha: гиперпараметр асимметрии
        lr_weights: learning rate для обновления весов
    """
    
    def __init__(self,
                 initial_lambda_graph: float = 1.0,
                 alpha: float = 1.5,
                 lr_weights: float = 0.01):
        super().__init__(initial_lambda_graph=initial_lambda_graph)
        self.alpha = alpha
        self.lr_weights = lr_weights

        self._log_weights = torch.nn.Parameter(
            torch.tensor([0.0, np.log(initial_lambda_graph)], dtype=torch.float64)
        )
        self._weights_optimizer = torch.optim.Adam([self._log_weights], lr=lr_weights)

        self._initial_model_loss: Optional[float] = None
        self._initial_graph_loss: Optional[float] = None

        self._last_grad_model: Optional[torch.Tensor] = None
        self._last_grad_graph: Optional[torch.Tensor] = None
    
    @property
    def name(self) -> str:
        return "gradnorm"
    
    def get_lambdas(self) -> Tuple[float, float]:
        with torch.no_grad():
            weights = torch.exp(self._log_weights)
            weights = weights / weights.sum() * 2
            return (weights[0].item(), weights[1].item())
    
    def compute_weighted_loss(self,
                              model_loss: torch.Tensor,
                              graph_loss: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет взвешенный loss для обратного распространения.
        Вызывается из trainer вместо простого lam_nn * model_loss + lam_graph * graph_loss
        """
        weights = torch.exp(self._log_weights)
        weights_normalized = weights / weights.sum() * 2
        return weights_normalized[0] * model_loss + weights_normalized[1] * graph_loss
    
    def store_gradients(self,
                        grad_model_norm: torch.Tensor,
                        grad_graph_norm: torch.Tensor) -> None:
        """
        Сохраняет нормы градиентов от model_loss и graph_loss.
        Вызывается из trainer после backward() для каждого loss отдельно.
        """
        self._last_grad_model = grad_model_norm.detach()
        self._last_grad_graph = grad_graph_norm.detach()
    
    def update(self,
               epoch: int,
               model_loss: float,
               graph_loss: float,
               model: Optional[nn.Module] = None,
               **kwargs) -> None:
        
        self._record_history(model_loss, graph_loss)

        if self._initial_model_loss is None:
            self._initial_model_loss = model_loss
            self._initial_graph_loss = graph_loss
            return

        if self._last_grad_model is None or self._last_grad_graph is None:
            self._update_simple(model_loss, graph_loss)
            return
        
        # GradNorm update
        self._update_gradnorm(model_loss, graph_loss)
    
    def _update_simple(self, model_loss: float, graph_loss: float) -> None:

        r_model = model_loss / (self._initial_model_loss + 1e-10)
        r_graph = graph_loss / (self._initial_graph_loss + 1e-10)

        r_mean = (r_model + r_graph) / 2

        target_model = (r_model / r_mean) ** self.alpha
        target_graph = (r_graph / r_mean) ** self.alpha

        with torch.no_grad():
            current_weights = torch.exp(self._log_weights)
            target_weights = torch.tensor([target_model, target_graph], dtype=torch.float64)
            target_weights = target_weights / target_weights.sum() * 2

            new_weights = current_weights + self.lr_weights * (target_weights - current_weights)
            self._log_weights.data = torch.log(new_weights + 1e-10)
    
    def _update_gradnorm(self, model_loss: float, graph_loss: float) -> None:
        """Полный GradNorm update с градиентами"""

        r_model = model_loss / (self._initial_model_loss + 1e-10)
        r_graph = graph_loss / (self._initial_graph_loss + 1e-10)
        r_mean = (r_model + r_graph) / 2

        weights = torch.exp(self._log_weights)
        G_model = self._last_grad_model * weights[0]
        G_graph = self._last_grad_graph * weights[1]
        G_mean = (G_model + G_graph) / 2
        
        target_model = G_mean * (r_model / r_mean) ** self.alpha
        target_graph = G_mean * (r_graph / r_mean) ** self.alpha
        

        gradnorm_loss = (torch.abs(G_model - target_model) + 
                         torch.abs(G_graph - target_graph))

        self._weights_optimizer.zero_grad()
        gradnorm_loss.backward()
        self._weights_optimizer.step()

        with torch.no_grad():
            weights = torch.exp(self._log_weights)
            weights = weights / weights.sum() * 2
            self._log_weights.data = torch.log(weights + 1e-10)

        self._last_grad_model = None
        self._last_grad_graph = None
    
    def reset(self) -> None:
        super().reset()
        self._log_weights = torch.nn.Parameter(
            torch.tensor([0.0, np.log(self.initial_lambda_graph)], dtype=torch.float64)
        )
        self._weights_optimizer = torch.optim.Adam([self._log_weights], lr=self.lr_weights)
        self._initial_model_loss = None
        self._initial_graph_loss = None
        self._last_grad_model = None
        self._last_grad_graph = None


class UncertaintyWeightingOptimizer(BaseLambdaOptimizer):
    """
    Uncertainty Weighting
    Reference: Kendall et al., "Multi-task Learning Using Uncertainty to Weigh 
               Losses for Scene Geometry and Semantics", CVPR 2018
               https://arxiv.org/abs/1705.07115
    
    Loss = (1/(2σ₁²)) * L_model + (1/(2σ₂²)) * L_graph + log(σ₁) + log(σ₂)

    """
    
    def __init__(self,
                 initial_lambda_graph: float = 1.0,
                 lr_sigma: float = 0.01):
        super().__init__(initial_lambda_graph=initial_lambda_graph)
        self.lr_sigma = lr_sigma
        
        # Обучаемые log(σ) для численной стабильности
        # Инициализация: σ=1 для model, σ=1/sqrt(2*lambda) для graph
        initial_log_sigma_graph = -0.5 * np.log(2 * initial_lambda_graph + 1e-10)
        self._log_sigmas = torch.nn.Parameter(
            torch.tensor([0.0, initial_log_sigma_graph], dtype=torch.float64)
        )
        self._sigma_optimizer = torch.optim.Adam([self._log_sigmas], lr=lr_sigma)
    
    @property
    def name(self) -> str:
        return "uncertainty_weighting"
    
    def get_lambdas(self) -> Tuple[float, float]:
        """
        Возвращает веса в формате (lam_nn, lam_graph).
        
        weight = 1 / (2 * σ²) = 0.5 * exp(-2 * log_sigma)
        """
        with torch.no_grad():
            sigmas_sq = torch.exp(2 * self._log_sigmas)
            weights = 0.5 / sigmas_sq
            return (weights[0].item(), weights[1].item())
    
    def compute_weighted_loss(self,
                              model_loss: torch.Tensor,
                              graph_loss: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет полный uncertainty-weighted loss включая регуляризацию.
        
        L = (1/(2σ₁²)) * L_model + (1/(2σ₂²)) * L_graph + log(σ₁) + log(σ₂)
        """
        sigmas_sq = torch.exp(2 * self._log_sigmas)
        
        weighted_model = 0.5 / sigmas_sq[0] * model_loss
        weighted_graph = 0.5 / sigmas_sq[1] * graph_loss

        regularization = self._log_sigmas.sum()
        
        return weighted_model + weighted_graph + regularization
    
    def update(self,
               epoch: int,
               model_loss: float,
               graph_loss: float,
               model: Optional[nn.Module] = None,
               **kwargs) -> None:

        self._record_history(model_loss, graph_loss)

        L_model = torch.tensor(model_loss, dtype=torch.float64)
        L_graph = torch.tensor(graph_loss, dtype=torch.float64)
        

        loss = self.compute_weighted_loss(L_model, L_graph)

        self._sigma_optimizer.zero_grad()
        loss.backward()
        self._sigma_optimizer.step()
    
    def update_with_backprop(self,
                             model_loss: torch.Tensor,
                             graph_loss: torch.Tensor) -> torch.Tensor:

        return self.compute_weighted_loss(model_loss, graph_loss)
    
    def reset(self) -> None:
        super().reset()
        initial_log_sigma_graph = -0.5 * np.log(2 * self.initial_lambda_graph + 1e-10)
        self._log_sigmas = torch.nn.Parameter(
            torch.tensor([0.0, initial_log_sigma_graph], dtype=torch.float64)
        )
        self._sigma_optimizer = torch.optim.Adam([self._log_sigmas], lr=self.lr_sigma)



def create_optimizer(method: str, **kwargs) -> BaseLambdaOptimizer:

    method = method.lower()
    
    if method == 'baseline':
        lambda_graph = kwargs.get('lambda_graph', 0.0)
        return FixedLambdaOptimizer(lambda_graph=lambda_graph)
    
    elif method == 'sobol':
        return SobolLambdaOptimizer(
            initial_lambda_graph=kwargs.get('initial_lambda_graph', 1.0),
            warmup_fraction=kwargs.get('warmup_fraction', 0.1),
            n_samples=kwargs.get('n_samples', 5)
        )
    
    elif method == 'gradnorm':
        return GradNormOptimizer(
            initial_lambda_graph=kwargs.get('initial_lambda_graph', 1.0),
            alpha=kwargs.get('alpha', 1.5),
            lr_weights=kwargs.get('lr_weights', 0.01)
        )
    
    elif method == 'uncertainty_weighting':
        return UncertaintyWeightingOptimizer(
            initial_lambda_graph=kwargs.get('initial_lambda_graph', 1.0),
            lr_sigma=kwargs.get('lr_sigma', 0.01)
        )
    
    else:
        raise ValueError(f"Unknown optimizer method: {method}. "
                        f"Available: baseline, sobol, gradnorm, uncertainty_weighting")
