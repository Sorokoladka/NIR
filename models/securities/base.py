from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

class SecurityModel(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def get_weight(self) -> float:
        pass

    @abstractmethod
    def simulate_path(self, correlated_shocks: np.ndarray, n_years: float, dt: float = 1/252) -> np.ndarray:
        pass

@dataclass
class SecurityParams:
    """Базовые параметры любого актива."""
    weight: float
    sigma: float  # волатильность

@dataclass
class StockParams(SecurityParams):
    mu: float  # ожидаемая доходность

@dataclass
class BondParams(SecurityParams):
    a: float   # скорость возврата к среднему
    b: float   # долгосрочное среднее
    Y0: float  # начальная доходность