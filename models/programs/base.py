from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, Union
import numpy as np
import pandas as pd
from models.macro.base import BaseSalaryModel, BaseUnemploymentModel

from models.macro.salary import DeterministicSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel
from quality.metrics import savings_metric, roi_metric, irr_metric, twr_metric, kz_metric, pension_metric

@dataclass
class ProgramInput:
    n: int
    age: int
    sex: Literal['M', 'F']
    rates: Optional[list[float]] = None

    # Взносы
    payment_mode: Literal['const', 'relative'] = 'const'
    payment_rate: float = None
    fix_payment: float = None

    # Макромодели (плагины)
    salary_model: Optional[BaseSalaryModel] = None
    unemployment_model: Optional[BaseUnemploymentModel] = None

    # Для совместимости: можно передать и скаляры
    initial_salary: Optional[Union[int, float]] = None

    # Налоговый вычет
    tax_deduction_rate: float = 0


class BaseProgram(ABC):
    def __init__(self, params: ProgramInput, life_table):
        self.params = params
        self.life_table = life_table
        self.tax = self.params.tax_deduction_rate

        self.__check_input__()

        # Инициализация атрибутов
        self.shocks = None
        self.annual_salaries = None
        self.payments = None
        self.tax_deduction = None
        self.co_financing = None
        self.total_inflows = None
        self.portfolio_path = None
        self.final_accumulation = None
        self.first_pension = None
        self.kz = None

    def __check_input__(self):

        if (self.params.payment_mode == 'relative') & (self.params.payment_rate is None):
            raise ValueError('Determine payment_rate for relative payment_mode')

        if (self.params.payment_mode == 'const') & (self.params.fix_payment is None):
            raise ValueError('Determine fix_payment for const payment_mode')

    def run(self):
        self._simulate_salary()
        self._apply_unemployment_shocks()
        self._calculate_contributions()
        self._apply_co_financing()
        self._accumulate_with_fees()
        self._finalize()
        return self

    # === Макромодели ===

    def _simulate_salary(self):
        if self.params.salary_model is not None:
            self.annual_salaries = self.params.salary_model.simulate(
                n_years=self.params.n,
                initial_salary=self.params.initial_salary
            )
        else:
            self.annual_salaries = np.full(self.params.n, self.params.initial_salary)

    def _apply_unemployment_shocks(self):
        if self.params.unemployment_model is not None:
            self.shocks = self.params.unemployment_model.simulate_shocks(self.params.n)
            self.annual_salaries *= self.shocks
        else:
            self.shocks = np.ones(self.params.n + 1)

    # === Взносы ===

    def _calculate_contributions(self):
        if self.params.payment_mode == 'relative':
            self._calculate_salary_contributions()
        else:
            self._calculate_fixed_contributions()

    def _calculate_salary_contributions(self):
        n = self.params.n
        payments = np.zeros(n)
        payments[0] = self.annual_salaries[0] * self.params.payment_rate
        tax_deduction = np.zeros(n)

        for i in range(1, n):
            tax_deduction[i] = payments[i-1] * self.tax
            payments[i] = self.annual_salaries[i] * self.params.payment_rate + tax_deduction[i]

        tax_deduction = np.append(tax_deduction, payments[-1] * self.tax)
        payments = np.append(payments, 0.0)
        self.payments = payments
        self.tax_deduction = tax_deduction

    def _calculate_fixed_contributions(self):
        n = self.params.n
        base_payments = np.array([self.params.fix_payment] * n + [0.0])
        tax_deduction = np.array([0.0] + [self.params.fix_payment * self.tax] * n)

        base_payments *= self.shocks
        tax_deduction *= self.shocks

        self.payments = base_payments
        self.tax_deduction = tax_deduction

    # === Софинансирование ===

    def _apply_co_financing(self):
        self.co_financing = np.zeros(self.params.n + 1)

    # === Накопление ===

    def _accumulate_with_fees(self):
        self.total_inflows = self.payments + self.tax_deduction + self.co_financing
        portfolio = self._simulate_portfolio_with_fees(self.total_inflows, self.params.rates)
        self.portfolio_path = portfolio
        self.final_accumulation = portfolio[-1]

    def _simulate_portfolio_with_fees(self, total_inflows: np.ndarray, rates: list[float]) -> np.ndarray:
        n = len(rates)
        do = np.zeros(n)
        after_fee = np.zeros(n)

        # Шаг 0
        do[0] = total_inflows[0]
        perc0 = do[0] * rates[0]
        after0 = do[0] + perc0
        fee0 = self._calculate_fee(perc0, 0, after0)
        after_fee[0] = after0 - fee0

        # Шаги 1..n-1
        for i in range(1, n):
            do[i] = total_inflows[i] + after_fee[i-1]
            perc = do[i] * rates[i]
            after_i = do[i] + perc
            fee = self._calculate_fee(perc, after_fee[i-1], after_i)
            after_fee[i] = after_i - fee

        return np.append(after_fee, after_fee[-1] + self.tax_deduction[-1])

    def _calculate_fee(self, perc, prev_value, current_value):
        return 0

    def compute_metrics(self) -> dict[str, float]:

        n = self.params.n
        total_payments = self.payments[:-1].sum()

        annuity_months = None
        if self.life_table is not None:
            annuity_months = self.life_table.loc[self.params.age + n, self.params.sex]

        pension = pension_metric(self.final_accumulation, annuity_months)

        result = {}

        result['savings'] = savings_metric(self.final_accumulation)
        result['roi'] = roi_metric(self.final_accumulation, total_payments)
        result['irr'] = irr_metric(self.payments, self.final_accumulation)
        result['twr'] = twr_metric(self.portfolio_path, self.payments)
        result['pension'] = pension
        result['kz'] = kz_metric(pension, self.annual_salaries, n)

        return result

    def _finalize(self) -> None:

        if self.life_table is not None:
            annuity_months = self.life_table.loc[self.params.age + self.params.n, self.params.sex]
            self.first_pension = self.final_accumulation / annuity_months
        else:
            self.first_pension = None

    def get_detailed_report(self) -> pd.DataFrame:
        if self.portfolio_path is None:
            raise ValueError("Call .run() first")

        n = self.params.n
        years = np.arange(1, n + 2)

        monthly_salary = np.full(n + 1, np.nan)
        annual_salary = np.full(n + 1, np.nan)
        shocks = np.full(n + 1, np.nan)
        payments = np.full(n + 1, np.nan)
        tax_deduction = np.full(n + 1, np.nan)
        co_financing = np.full(n + 1, np.nan)
        total_inflows = np.full(n + 1, np.nan)
        do = np.full(n + 1, np.nan)
        rates_pct = np.full(n + 1, np.nan)
        perc = np.full(n + 1, np.nan)
        var_fee = np.full(n + 1, np.nan)
        fixed_fee = np.full(n + 1, np.nan)
        total_fee = np.full(n + 1, np.nan)
        after_fee = np.full(n + 1, np.nan)

        monthly_salary[:n] = self.annual_salaries[:n] / 12
        annual_salary[:n] = self.annual_salaries[:n]
        shocks[:n + 1] = self.shocks
        payments[:n + 1] = self.payments
        tax_deduction[:n + 1] = self.tax_deduction
        co_financing[:n + 1] = self.co_financing
        total_inflows[:n + 1] = self.total_inflows

        for i in range(n):
            do[i] = total_inflows[i]
            rate = self.params.rates[i]
            rates_pct[i] = rate * 100
            perc[i] = do[i] * rate

            after_before_fee = do[i] + perc[i]
            if i == 0:
                prev_val = 0.0
            else:
                prev_val = after_fee[i - 1]

            var_component = getattr(self, 'var_rate', 0.0) * perc[i]
            fixed_component = getattr(self, 'fixed_rate', 0.0) * (prev_val + after_before_fee) / 2

            var_fee[i] = var_component
            fixed_fee[i] = fixed_component
            total_fee[i] = var_component + fixed_component
            after_fee[i] = after_before_fee - total_fee[i]

        after_fee[n] = self.final_accumulation

        df = pd.DataFrame({
            'Год': years,
            'Зарплата (месячная)': monthly_salary,
            'Зарплата (годовая)': annual_salary,
            'Шок занятости': shocks,
            'Взнос вкладчика': payments,
            'Налоговый вычет': tax_deduction,
            'Софинансирование': co_financing,
            'Итоговый взнос в год': total_inflows,
            'Баланс до начисления %': do,
            'Годовая ставка (%)': rates_pct,
            'Начисленный доход (руб)': perc,
            'Комиссия: переменная': var_fee,
            'Комиссия: фиксированная': fixed_fee,
            'Итого комиссия': total_fee,
            'Баланс после комиссии': after_fee
        })

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(2)

        return df