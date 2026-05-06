"""
Гипотеза H1
-----------
Программы с самостоятельным управлением счётом (ИИС-3) имеют более высокую
ожидаемую доходность, чем ПДС, за счёт менее ограниченных требований к структуре
и составу портфеля.

Дизайн эксперимента
-------------------
Переменная идентификации: portfolio_structure (4 варианта — ПДС + 3 ИИС-3)
Метрики:
  • TWR  — time-weighted return, изолирует портфельную доходность от взносов
  • IRR  — полная программная доходность с учётом комиссий и софинансирования
  • ROI  — накопленная доходность на собственные взносы
Контроль: salary × sex × age (фиксированы)

Структура ИИС-3
  Облигационная часть делится в пропорциях среднего портфеля НПФ:
    gov_bond ≈ 63.5 %, corp_bond ≈ 33.4 %, mun_bond ≈ 3.1 %
  Три аллокации акции/облигации: 20/80, 50/50, 80/20

Интерпретация
  Если E[TWR_ИИС] > E[TWR_ПДС] — гипотеза подтверждается на уровне портфельной
  доходности. Если E[IRR_ИИС] < E[IRR_ПДС] при E[TWR_ИИС] > E[TWR_ПДС], это
  свидетельствует, что комиссии/со-финансирование ПДС перевешивают портфельное
  преимущество ИИС-3 при данном уровне взносов.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from models.securities.portfolio import PortfolioModel
from models.programs.base import ProgramInput
from models.programs.pds import PDSProgram
from models.programs.iis3 import IIS3Program
from models.macro.salary import StochasticSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel

from scenarios import (indexes, structure, life_table, YIELD_COL,
                       unemployment_k, unemployment_p, unemployment_lambda)
from models.securities import get_security_params

# ── параметры эксперимента ────────────────────────────────────────────────────
N_SIMULATIONS = 500
N_YEARS       = 15
PAYMENT_RATE  = 0.06          # 6 % от зарплаты
TAX_RATE      = 0.13

SALARY_RANGE = [50_000, 100_000, 150_000, 200_000]   # покрывает все 3 тира со-фин.
AGE_RANGE    = [30, 45]
SEX_RANGE    = ['M', 'F']

# ── порядок активов должен совпадать с порядком YIELD-колонок в indexes ───────
ASSET_ORDER  = ['stock', 'gov_bond', 'corp_bond', 'mun_bond']
YIELD_COLS   = [f'{YIELD_COL}_{a}' for a in ASSET_ORDER]
CORR_4X4     = indexes[YIELD_COLS].corr().values   # 4×4

# ── пропорции облигационной части из среднего портфеля НПФ ───────────────────
pds_avg = structure.iloc[-1]
_bond_total = pds_avg[['gov_bond', 'corp_bond', 'mun_bond']].sum()
BOND_RATIOS = {
    'gov_bond':  pds_avg['gov_bond']  / _bond_total,
    'corp_bond': pds_avg['corp_bond'] / _bond_total,
    'mun_bond':  pds_avg['mun_bond']  / _bond_total,
}


def build_iis3_structure(stock_share: float) -> dict:
    """
    Строит словарь весов для ИИС-3 с заданной долей акций.
    Облигационная часть делится пропорционально среднему НПФ.
    Порядок ключей: stock → gov_bond → corp_bond → mun_bond (= ASSET_ORDER).
    """
    bond_share = 1.0 - stock_share
    return {
        'stock':     stock_share,
        'gov_bond':  bond_share * BOND_RATIOS['gov_bond'],
        'corp_bond': bond_share * BOND_RATIOS['corp_bond'],
        'mun_bond':  bond_share * BOND_RATIOS['mun_bond'],
    }


# ── портфели ─────────────────────────────────────────────────────────────────
PORTFOLIOS = {
    'pds_avg':    structure.iloc[-1][ASSET_ORDER].to_dict(),
    'iis3_20/80': build_iis3_structure(0.20),
    'iis3_50/50': build_iis3_structure(0.50),
    'iis3_80/20': build_iis3_structure(0.80),
}

# ── симулируем доходности один раз для каждого портфеля ──────────────────────
if __name__ == '__main__':

    print("Симулирую портфели...")
    simulated_returns = {}
    for label, struct_dict in PORTFOLIOS.items():
        securities = get_security_params(indexes=indexes, structure=struct_dict)
        portfolio  = PortfolioModel(assets=securities, corr_matrix=CORR_4X4)
        simulated_returns[label] = portfolio.simulate(
            n_years=N_YEARS, n_simulations=N_SIMULATIONS, dt=1/252, show_progress=False
        )

    # ── безработица (общая для всех сценариев) ────────────────────────────────
    unemployment_model = WeibullUnemploymentModel(
        p_exit=unemployment_p, weibull_k=unemployment_k, weibull_lambda=unemployment_lambda
    )

    # ── основной цикл ─────────────────────────────────────────────────────────
    rows = []

    for salary in tqdm(SALARY_RANGE, desc='salary'):
        for age in AGE_RANGE:
            salary_model = StochasticSalaryModel(initial_age=age)
            for sex in SEX_RANGE:
                for portfolio_label, returns_matrix in simulated_returns.items():
                    is_pds = portfolio_label.startswith('pds')

                    for i in range(N_SIMULATIONS):
                        rates = list(returns_matrix[:, i][::252])
                        n     = len(rates)

                        params = ProgramInput(
                            n=n, age=age, sex=sex,
                            rates=rates,
                            payment_mode='relative',
                            payment_rate=PAYMENT_RATE,
                            initial_salary=salary,
                            tax_deduction_rate=TAX_RATE,
                            salary_model=salary_model,
                            unemployment_model=unemployment_model,
                        )

                        prog = (PDSProgram(params=params, life_table=life_table)
                                if is_pds
                                else IIS3Program(params=params, life_table=life_table))
                        prog.run()
                        metrics = prog.compute_metrics()

                        rows.append({
                            'program':   'pds'  if is_pds else 'iis3',
                            'portfolio': portfolio_label,
                            'salary':    salary,
                            'age':       age,
                            'sex':       sex,
                            'sim_id':    i,
                            **metrics,
                        })

    df = pd.DataFrame(rows)
    df.to_csv('temp_data/h1_portfolio_freedom.csv', index=False)
    print(f"Сохранено: temp_data/h1_portfolio_freedom.csv  ({len(df)} строк)")
