import numpy as np
from .base import SecurityModel, BondParams

class BondModel(SecurityModel):
    def __init__(self, params: BondParams):
        super().__init__(params)

    def get_weight(self) -> float:
        return self.params.weight

    def simulate_path(self, correlated_shocks: np.ndarray, n_years: float, dt: float = 1/252) -> np.ndarray:
        n_steps = int(n_years / dt)
        Y = np.zeros(n_steps)
        Y[0] = self.params.Y0
        for t in range(1, n_steps):
            Y[t] = (Y[t-1] + self.params.a * (self.params.b - Y[t-1]) * dt +
                    self.params.sigma * np.sqrt(dt) * correlated_shocks[t])
        return Y
