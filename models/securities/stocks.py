import numpy as np
import pandas as pd
from .base import SecurityModel, StockParams

class StockModel(SecurityModel):
    def __init__(self, params: StockParams):
        super().__init__(params)

    def get_weight(self) -> float:
        return self.params.weight

    def simulate_path(self, correlated_shocks: np.ndarray, n_years: float, dt: float = 1/252) -> np.ndarray:
        n_steps = int(n_years / dt)
        returns = self.params.mu * dt + self.params.sigma * np.sqrt(dt) * correlated_shocks[:n_steps]
        series = pd.Series(1 + returns)
        rolling = series.rolling(window=252).apply(lambda x: x.prod()) - 1
        return np.nan_to_num(rolling.values, nan=0.0)