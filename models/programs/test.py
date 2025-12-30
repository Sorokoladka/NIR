from .base import BaseProgram, ProgramInput
import numpy as np


class PDS2Program(BaseProgram):
    def __init__(self, params: ProgramInput, life_table):
        super().__init__(params, life_table)

        self.var_rate = 0.2
        self.fixed_rate = 0.005

    def _apply_co_financing(self) -> None:
        n = self.params.n
        co_fin = np.zeros(n + 1)
        initial_salary = self.params.initial_salary

        member_contrib = self.payments[:n]

        for i in range(1, min(11, n + 1)):
            contrib_prev_year = member_contrib[i - 1]

            if initial_salary < 80_000:
                co_fin[i] = min(36_000, contrib_prev_year)
            elif initial_salary < 150_000:
                co_fin[i] = min(36_000, contrib_prev_year / 2)
            else:
                co_fin[i] = min(36_000, contrib_prev_year / 4)

        self.co_financing = co_fin

    def _calculate_fee(self, perc: float, prev_value: float, current_value: float) -> float:
        var_fee = self.var_rate * perc
        fixed_fee = self.fixed_rate * (prev_value + current_value) / 2
        return var_fee + fixed_fee