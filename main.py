import numpy as np

from models.securities.portfolio import PortfolioModel
from models.securities.bonds import BondModel, BondParams
from models.securities.stocks import StockModel, StockParams

from models.programs.base import ProgramInput
from models.programs.pds import PDSProgram
from models.programs.iis3 import IIS3Program

from models.macro.salary import DeterministicSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel

from scenarios import indexes, structure, life_table, INDEXES, SECURITY_TYPES, YIELD_COL,\
    unemployment_k, unemployment_p, unemployment_lambda
from models.securities import get_security_params

if __name__ == '__main__':

    id = 122
    # 1. Модель для портфеля (доходности)
    securities = get_security_params(structure, indexes, id=id)
    corr_matrix = indexes[[col for col in indexes.columns if YIELD_COL in col]].corr()

    portfolio = PortfolioModel(assets=securities, corr_matrix=corr_matrix)
    simulated_returns = portfolio.simulate(n_years=15, n_simulations=1, dt=1/252)

############

    # 2. Макро модели

    salary_model = DeterministicSalaryModel(annual_growth=0.05) #or stochastic
    unemployment_model = WeibullUnemploymentModel( #or None
        p_exit = unemployment_p,
        weibull_k = unemployment_k,
        weibull_lambda = unemployment_lambda
    )

    # 3. Итерация программы
    params = ProgramInput(
        n = 15,
        age = 45,
        sex = 'M',
        rates = simulated_returns[:,0][::252],
        payment_mode = 'relative',
        payment_rate = 0.06,
        salary_model = salary_model,
        unemployment_model = unemployment_model,
        initial_salary = 50000,
        tax_deduction_rate = 0.13
    )

    program_calculator = PDSProgram(params=params, life_table=life_table)
    res1 = program_calculator.run()
    res2 = program_calculator.compute_metrics()
    res3 = program_calculator.get_detailed_report()

    print(1)