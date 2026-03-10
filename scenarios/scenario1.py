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

# СЦЕНАРИЙ 1
# Хочу оценить эффект затухания софинансирования
# Суть: государственное софинансирование в ПДС действует только 10 лет, а минимальный срок договора — 15 лет.
# Хочу оценить, «съедает» ли комиссия НПФ 20% от профита преимущество, накопленное за первые 10 лет софинансирования.

# Гипотеза: для участников со сроком участия >12 лет маржинальная полезность ПДС снижается в последние 3–5 лет
# действия договора из-за прекращения софинансирования при сохранении комиссий НПФ (0.5% + 20% от дохода).
# В длинных сценариях (15+ лет) ИИС-3 с реинвестированием дивидендов может обогнать ПДС по медианному NPV,
# несмотря на отсутствие софинансирования, за счет отсутствия комиссии за успех (performance fee).

# Как буду проверять:
# 1. задаю срок софинансирования интервалом 15-20 лет
# 2. в данном сценарии не фокусируюсь на демографии - беру случайный тип для исследования (М,45) с разным уровнем дохода (50, 100, 200)
# 3. для ПДС беру усредненную стратегию (соотношение а/о), для ИИС беру 3 варианта (20/80, 50/50, 80,20)
# 4. макро модели не применяем
# 5. строю по годам динамику NPV (всего 3 графика для каждой стратегии ИИС)

if __name__ == '__main__':

    # 0. debug
    n_simulations = 1000

    # 1. задаю срок софинансирования интервалом 15-20 лет
    n_range = range(15,21)
    salary_range = [50000, 100000, 200000]

    # 2. в данном сценарии не фокусируюсь на демографии - беру случайный тип для исследования (М,45)
    age = 45
    sex = 'M'

    # 3. формирую стратегии инвестирования
    simulated_returns = {}

    for n in n_range:

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
            simulated_returns[f'iis_{int(stock*100)}/{int(bond*100)}_{n}'] = portfolio.simulate(n_years=n, n_simulations=n_simulations, dt=1/252)

    # 4. макро модели не применяем

    # итерация программы
    result_dict = {}
    for scenario in list(simulated_returns.keys()):
        result_dict_temp = {}

        for salary in salary_range:
            for i in range(n_simulations):

                returns = simulated_returns[scenario][:,i][::252] #на конец каждого года
                n = len(returns)

                params = ProgramInput(
                    n = n,
                    age = 45,
                    sex = 'M',
                    rates = returns,
                    payment_mode = 'relative',
                    payment_rate = 0.06,
                    initial_salary = salary,
                    tax_deduction_rate = 0.13
                )

                if 'pds' in scenario:
                    program_calculator = PDSProgram(params=params, life_table=life_table)
                    program_calculator.run()

                    result_dict_temp[f'{salary}_{i}'] = program_calculator.compute_metrics()

                elif 'iis' in scenario:
                    program_calculator = IIS3Program(params=params, life_table=life_table)
                    program_calculator.run()

                    result_dict_temp[f'{salary}_{i}'] = program_calculator.compute_metrics()

        result_dict[scenario] = result_dict_temp

    # 5. собираем данные для анализа

    rows = [
        {
            'scenario_n': scenario,
            'salary_i': salary,
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
    pre_df['salary'] = pre_df['salary_i'].apply(lambda x: int(x.split('_')[0]))
    pre_df['i'] = pre_df['salary_i'].apply(lambda x: int(x.split('_')[-1]))
    df = pre_df.drop(columns=['scenario_n','salary_i'])

    df.to_csv('temp_data/scenario1.csv', index=False)

    print(1)