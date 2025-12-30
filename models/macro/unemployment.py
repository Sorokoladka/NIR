import numpy as np
from .base import BaseUnemploymentModel

class WeibullUnemploymentModel(BaseUnemploymentModel):
    def __init__(self, p_exit: float, weibull_k: float, weibull_lambda: float):
        self.p_exit = p_exit
        self.weibull_k = weibull_k
        self.weibull_lambda = weibull_lambda

    def _generate_duration(self, size: int) -> np.ndarray:
        U = np.random.uniform(0, 1, size=size)
        return self.weibull_lambda * (-np.log(U)) ** (1 / self.weibull_k)

    def simulate_shocks(self, n_years: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        shocks = np.ones(n_years + 1)
        binary = np.random.binomial(1, 1 - self.p_exit, size=n_years)
        shocks[1:] = binary

        durations = self._generate_duration(n_years + 1) / 12
        durations = np.minimum(1.0, durations)
        employment = 1.0 - durations
        shocks = np.where(shocks == 0, employment, shocks)
        return shocks
