import numpy as np
from tqdm import tqdm
from .base import SecurityModel

class PortfolioModel:
    def __init__(self, assets: list[SecurityModel], corr_matrix: np.ndarray):
        self.assets = assets
        self.corr_matrix = corr_matrix
        self.n_assets = len(assets)
        self.L = np.linalg.cholesky(corr_matrix)

        self.__check_weights__()

    def __check_weights__(self):
        total_weight = sum(a.get_weight() for a in self.assets)

        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weigts sum should be equal 1, receive: {total_weight}") #усреднять по всем фондам нельзя

    def simulate(self, n_years: int, n_simulations: int = 5, dt: float = 1/252, show_progress: bool = True):
        n_steps = int(n_years / dt)
        portfolio_values = np.zeros((n_steps, n_simulations))

        for sim in tqdm(range(n_simulations)):
            Z = np.random.normal(size=(self.n_assets, n_steps))
            Z_corr = self.L @ Z

            for i, asset in enumerate(self.assets):
                path = asset.simulate_path(Z_corr[i], n_years, dt)
                portfolio_values[:, sim] += path * asset.get_weight()

        return portfolio_values
