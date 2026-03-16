import numpy as np
from models.macro.base import BaseSalaryModel
from typing import Callable

class DeterministicSalaryModel(BaseSalaryModel):
    def __init__(self, annual_growth: float = 0.05):
        self.annual_growth = annual_growth

    def simulate(self, n_years: int, initial_salary: float) -> np.ndarray:
        salaries = np.zeros(n_years)
        salaries[0] = initial_salary
        for i in range(1, n_years):
            salaries[i] = salaries[i-1] * (1 + self.annual_growth)
        return np.append(salaries * 12, 0.0)

class StochasticSalaryModel(BaseSalaryModel):
    def __init__(
            self,
            initial_age: int,
            mu: float = 0.0086,
            rho: float = 0.6,
            ipc: float = 0.0726,
            u_func: Callable[[float], float] = lambda x: 10 * (x-8.8) ** 1.04 * np.exp((8.8 - x) / 28.1)
    ):
        self.initial_age = initial_age
        self.mu = mu
        self.rho = rho
        self.ipc = ipc
        self.u_func = u_func

    def simulate(self, n_years: int, initial_salary: float) -> np.ndarray:
        salaries = np.zeros(n_years)
        salaries[0] = initial_salary
        for i in range(1, n_years):
            growth = self.mu + self.rho * self.ipc + (self.u_func(self.initial_age+i) / self.u_func(self.initial_age+i-1) - 1)
            salaries[i] = salaries[i-1] * (1 + growth)
        return np.append(salaries * 12, 0.0)


if __name__ == '__main__':
    a = StochasticSalaryModel()
    b = DeterministicSalaryModel()

    resb = b.simulate(10, 10000)
    resa = a.simulate(10, 10000, 50)
    print(1)