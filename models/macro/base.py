from abc import ABC, abstractmethod
import numpy as np

class BaseSalaryModel(ABC):
    @abstractmethod
    def simulate(self, n_years: int, initial_salary: float) -> np.ndarray:
        """
        Возвращает массив годовых зарплат длиной n_years + 1 (последний = 0).
        """
        pass


class BaseUnemploymentModel(ABC):
    @abstractmethod
    def simulate_shocks(self, n_years: int) -> np.ndarray:
        """
        Возвращает массив множителей длиной n_years + 1.
        1 = полностью занят, <1 = частичная безработица.
        """
        pass