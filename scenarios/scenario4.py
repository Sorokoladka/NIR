import numpy as np
from tqdm import tqdm
import pandas as pd

from models.securities.portfolio import PortfolioModel
from models.securities.bonds import BondModel, BondParams
from models.securities.stocks import StockModel, StockParams

from models.programs.base import ProgramInput
from models.programs.pds import PDSProgram
from models.programs.iis3 import IIS3Program

from models.macro.salary import DeterministicSalaryModel, StochasticSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel

from scenarios import indexes, structure, life_table, INDEXES, SECURITY_TYPES, YIELD_COL,\
    unemployment_k, unemployment_p, unemployment_lambda
from models.securities import get_security_params

# СЦЕНАРИЙ 4
# сформируем портрет (initial_age+sex+initial_salary) с лучшими результатами по ПДС и ИИС3

if __name__ == '__main__':

    # 0. debug
    n_simulations = 1000

    # 1. задаю срок софинансирования интервалом и уровень зп
    n = 15
    salary_range = [50000, 100000, 150000]

    # 2. демография
    age_range = np.arange(20,60)
    sex_range = ['F','M']

    # 3. формирую стратегии инвестирования
    simulated_returns = {}

    # 3.1 для ПДС беру усредненную стратегию - закодировали как 0
    id = 0
    structure_dict = structure.iloc[0].to_dict()
    securities = get_security_params(indexes=indexes, structure=structure_dict)
    corr_matrix = indexes[[col for col in indexes.columns if YIELD_COL in col]].corr()

    portfolio = PortfolioModel(assets=securities, corr_matrix=corr_matrix)
    simulated_returns[f'pds_{n}'] = portfolio.simulate(n_years=n, n_simulations=n_simulations, dt=1/252)

    # 3.2 для ИИС беру 3 вариант (20/80, 50/50, 80,20)
    for (stock, bond) in [(0.2,0.8), (0.5,0.5), (0.8,0.2)]:

        structure_dict = {'stock':stock, 'gov_bond':bond}
        securities = get_security_params(indexes=indexes, structure=structure_dict)
        corr_matrix = indexes[[col for col in indexes.columns if (YIELD_COL in col) & ('corp' not in col) & ('mun' not in col)]].corr()

        portfolio = PortfolioModel(assets=securities, corr_matrix=corr_matrix)
        simulated_returns[f'iis-{int(stock*100)}/{int(bond*100)}_{n}'] = portfolio.simulate(n_years=n, n_simulations=n_simulations, dt=1/252)

    # 4. макро модели применяем

    unemployment_model = WeibullUnemploymentModel( #or None
        p_exit = unemployment_p,
        weibull_k = unemployment_k,
        weibull_lambda = unemployment_lambda
    )

    # итерация программы
    result_dict = {}
    for scenario in list(simulated_returns.keys()):
        result_dict_temp = {}

        for salary in salary_range:
            for age in age_range:

                salary_model = StochasticSalaryModel(
                    initial_age=age
                )

                for sex in sex_range:

                    for i in range(n_simulations):

                        returns = simulated_returns[scenario][:,i][::252]
                        n = len(returns)

                        params = ProgramInput(
                            n = n,
                            age = age,
                            sex = sex,
                            rates = returns,
                            payment_mode = 'relative',
                            payment_rate = 0.06,
                            initial_salary = salary,
                            tax_deduction_rate = 0.13,
                            salary_model=salary_model,
                            unemployment_model=unemployment_model
                        )

                        if 'pds' in scenario:
                            program_calculator = PDSProgram(params=params, life_table=life_table)
                            program_calculator.run()

                            result_dict_temp[f'{salary}_{age}_{sex}_{i}'] = program_calculator.compute_metrics()

                        elif 'iis' in scenario:
                            program_calculator = IIS3Program(params=params, life_table=life_table)
                            program_calculator.run()

                            result_dict_temp[f'{salary}_{age}_{sex}_{i}'] = program_calculator.compute_metrics()

        result_dict[scenario] = result_dict_temp

    # 5. собираем данные для анализа

    rows = [
        {
            'scenario_n': scenario,
            'salary_age_sex_i': salary,
            'metric_name': metric,
            'metric_value': value
        }
        for scenario, salaries in result_dict.items()
        for salary, metrics in salaries.items()
        for metric, value in metrics.items()
    ]

    pre_df = pd.DataFrame.from_records(rows)

    pre_df['scenario'] = pre_df['scenario_n'].apply(lambda x: x.split('_')[0])
    pre_df['n'] = pre_df['scenario_n'].apply(lambda x: int(x.split('_')[-1]))
    pre_df['salary'] = pre_df['salary_age_sex_i'].apply(lambda x: int(x.split('_')[0]))
    pre_df['age'] = pre_df['salary_age_sex_i'].apply(lambda x: int(x.split('_')[1]))
    pre_df['sex'] = pre_df['salary_age_sex_i'].apply(lambda x: x.split('_')[2])
    pre_df['i'] = pre_df['salary_age_sex_i'].apply(lambda x: int(x.split('_')[-1]))
    df = pre_df.drop(columns=['scenario_n','salary_age_sex_i'])

    df.to_csv('temp_data/scenario4.csv', index=False)

    print(1)