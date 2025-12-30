import numpy as np
from .base import BaseSalaryModel

class DeterministicSalaryModel(BaseSalaryModel):
    def __init__(self, annual_growth: float = 0.05):
        self.annual_growth = annual_growth

    def simulate(self, n_years: int, initial_salary: float) -> np.ndarray:
        salaries = np.zeros(n_years)
        salaries[0] = initial_salary
        for i in range(1, n_years):
            salaries[i] = salaries[i-1] * (1 + self.annual_growth)
        return np.append(salaries * 12, 0.0)