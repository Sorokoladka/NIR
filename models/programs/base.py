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

    # взносы
    payment_mode: Literal['const', 'relative'] = 'const'
    payment_rate: float = None
    fix_payment: float = None
    initial_salary: Optional[Union[int, float]] = None

    # макромодели (плагины)
    salary_model: Optional[BaseSalaryModel] = None
    unemployment_model: Optional[BaseUnemploymentModel] = None

    # налоговый вычет
    tax_deduction_rate: float = 0


class BaseProgram(ABC):
    def __init__(self, params: ProgramInput, life_table):
        self.params = params
        self.life_table = life_table
        self.tax = self.params.tax_deduction_rate

        self.__check_input__()

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
        self._sim_details = None

    def __check_input__(self):
        if (self.params.payment_mode == 'relative') and (self.params.payment_rate is None):
            raise ValueError('Determine payment_rate for relative payment_mode')

        if (self.params.payment_mode == 'const') and (self.params.fix_payment is None):
            raise ValueError('Determine fix_payment for const payment_mode')

    def run(self):
        self._simulate_salary()
        self._apply_unemployment_shocks()
        self._calculate_contributions()
        self._apply_co_financing()
        self._accumulate_with_fees()
        self._finalize()
        return self

    def _simulate_salary(self):
        if self.params.salary_model is not None:
            self.annual_salaries = self.params.salary_model.simulate(
                n_years=self.params.n,
                initial_salary=self.params.initial_salary
            )
        else:
            self.annual_salaries = np.full(self.params.n + 1, float(self.params.initial_salary) * 12)

    def _apply_unemployment_shocks(self):
        if self.params.unemployment_model is not None:
            self.shocks = self.params.unemployment_model.simulate_shocks(self.params.n)
            self.annual_salaries *= self.shocks
        else:
            self.shocks = np.ones(self.params.n + 1)

    def _calculate_contributions(self):
        if self.params.payment_mode == 'relative':
            self._calculate_salary_contributions()
        else:
            self._calculate_fixed_contributions()

    def _calculate_salary_contributions(self):
        n = self.params.n
        payments = np.zeros(n)
        tax_deduction = np.zeros(n)

        payments[0] = self.annual_salaries[0] * self.params.payment_rate

        for i in range(1, n):
            tax_deduction[i] = payments[i-1] * self.tax
            payments[i] = self.annual_salaries[i] * self.params.payment_rate  # без вычета — он идёт отдельным потоком

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

    def _apply_co_financing(self):
        self.co_financing = np.zeros(self.params.n + 1)

    def _accumulate_with_fees(self):
        self.total_inflows = self.payments + self.tax_deduction + self.co_financing
        self._sim_details = self._simulate_portfolio_with_fees(self.total_inflows, self.params.rates)
        self.portfolio_path = self._sim_details['after_fee_path']
        self.final_accumulation = self.portfolio_path[-1]

    def _simulate_portfolio_with_fees(self, total_inflows: np.ndarray, rates: list[float]) -> dict:
        n = len(rates)
        do = np.zeros(n)
        perc_arr = np.zeros(n)
        var_fee_arr = np.zeros(n)
        fixed_fee_arr = np.zeros(n)
        total_fee_arr = np.zeros(n)
        after_fee = np.zeros(n)

        do[0] = total_inflows[0]
        perc_arr[0] = do[0] * rates[0]
        after0 = do[0] + perc_arr[0]
        total_fee_arr[0], var_fee_arr[0], fixed_fee_arr[0] = self._calculate_fee(perc_arr[0], 0.0, after0)
        after_fee[0] = after0 - total_fee_arr[0]

        for i in range(1, n):
            do[i] = total_inflows[i] + after_fee[i-1]
            perc_arr[i] = do[i] * rates[i]
            after_i = do[i] + perc_arr[i]
            total_fee_arr[i], var_fee_arr[i], fixed_fee_arr[i] = self._calculate_fee(perc_arr[i], after_fee[i-1], after_i)
            after_fee[i] = after_i - total_fee_arr[i]

        return {
            'do': do,
            'perc': perc_arr,
            'var_fee': var_fee_arr,
            'fixed_fee': fixed_fee_arr,
            'total_fee': total_fee_arr,
            'after_fee': after_fee,
            'after_fee_path': np.append(after_fee, after_fee[-1] + self.tax_deduction[-1]),
        }

    def _calculate_fee(self, perc, prev_value, current_value) -> tuple[float, float, float]:
        return 0.0, 0.0, 0.0  # total, var, fixed

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
            raise ValueError("call .run() first")

        n = self.params.n
        d = self._sim_details

        monthly_salary = np.full(n + 1, np.nan)
        annual_salary = np.full(n + 1, np.nan)
        monthly_salary[:n] = self.annual_salaries[:n] / 12
        annual_salary[:n] = self.annual_salaries[:n]

        do_col = np.full(n + 1, np.nan)
        do_col[:n] = d['do']
        rates_pct_col = np.full(n + 1, np.nan)
        rates_pct_col[:n] = np.array(self.params.rates) * 100
        perc_col = np.full(n + 1, np.nan)
        perc_col[:n] = d['perc']
        var_fee_col = np.full(n + 1, np.nan)
        var_fee_col[:n] = d['var_fee']
        fixed_fee_col = np.full(n + 1, np.nan)
        fixed_fee_col[:n] = d['fixed_fee']
        total_fee_col = np.full(n + 1, np.nan)
        total_fee_col[:n] = d['total_fee']
        after_fee_col = np.append(d['after_fee'], self.final_accumulation)

        df = pd.DataFrame({
            'Год': np.arange(1, n + 2),
            'Зарплата (месячная)': monthly_salary,
            'Зарплата (годовая)': annual_salary,
            'Шок занятости': self.shocks,
            'Взнос вкладчика': self.payments,
            'Налоговый вычет': self.tax_deduction,
            'Софинансирование': self.co_financing,
            'Итоговый взнос в год': self.total_inflows,
            'Баланс до начисления %': do_col,
            'Годовая ставка (%)': rates_pct_col,
            'Начисленный доход (руб)': perc_col,
            'Комиссия: переменная': var_fee_col,
            'Комиссия: фиксированная': fixed_fee_col,
            'Итого комиссия': total_fee_col,
            'Баланс после комиссии': after_fee_col
        })

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(2)

        return df
