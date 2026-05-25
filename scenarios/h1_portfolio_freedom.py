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

Сценарии вероятности трудового перехода
  baseline:    p_transition = calibrated (Weibull, ОРС Росстат 2024) — базовый
  low_transit: p_transition = 0.10 — низкая вероятность трудового перехода
  mid_transit: p_transition = 0.15 — умеренная вероятность трудового перехода

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
from scenarios.market_scenarios import MARKET_SCENARIOS, build_portfolio_for_scenario
import os

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_data')

# ── параметры эксперимента ────────────────────────────────────────────────────
N_SIMULATIONS = 5000
N_YEARS       = 15
PAYMENT_RATE  = 0.06          # 6 % от зарплаты
TAX_RATE      = 0.13

SALARY_RANGE = [50_000, 100_000, 150_000, 200_000]
AGE_RANGE    = [20, 40, 60]
SEX_RANGE    = ['M', 'F']

# ── сценарии вероятности трудового перехода ───────────────────────────────────
# "baseline" использует откалиброванное значение из ОРС Росстат 2024
# low_transit / mid_transit — альтернативные сценарии для проверки устойчивости
TRANSITION_SCENARIOS = {
    'baseline':    unemployment_p,   # откалиброванное значение
    'low_transit': 0.10,
    'mid_transit': 0.15,
}

# ── порядок активов должен совпадать с порядком YIELD-колонок в indexes ───────
ASSET_ORDER  = ['stock', 'gov_bond', 'corp_bond', 'mun_bond']
YIELD_COLS   = [f'{YIELD_COL}_{a}' for a in ASSET_ORDER]
CORR_4X4     = indexes[YIELD_COLS].corr().values

pds_avg = structure.iloc[-1]
_bond_total = pds_avg[['gov_bond', 'corp_bond', 'mun_bond']].sum()
BOND_RATIOS = {
    'gov_bond':  pds_avg['gov_bond']  / _bond_total,
    'corp_bond': pds_avg['corp_bond'] / _bond_total,
    'mun_bond':  pds_avg['mun_bond']  / _bond_total,
}


def build_iis3_structure(stock_share: float) -> dict:
    bond_share = 1.0 - stock_share
    return {
        'stock':     stock_share,
        'gov_bond':  bond_share * BOND_RATIOS['gov_bond'],
        'corp_bond': bond_share * BOND_RATIOS['corp_bond'],
        'mun_bond':  bond_share * BOND_RATIOS['mun_bond'],
    }


PORTFOLIOS = {
    'pds_avg':    structure.iloc[-1][ASSET_ORDER].to_dict(),
    'iis3_20/80': build_iis3_structure(0.20),
    'iis3_50/50': build_iis3_structure(0.50),
    'iis3_80/20': build_iis3_structure(0.80),
}


if __name__ == '__main__':

    rows = []

    for market_scenario in MARKET_SCENARIOS:
        print(f"Симулирую портфели [{market_scenario}]...")
        simulated_returns = {}
        for label, struct_dict in PORTFOLIOS.items():
            simulated_returns[label] = build_portfolio_for_scenario(
                scenario=market_scenario,
                indexes=indexes,
                structure=struct_dict,
                corr_matrix=CORR_4X4,
                n_years=N_YEARS,
                n_simulations=N_SIMULATIONS,
                yield_col=YIELD_COL,
                show_progress=False,
            )

        for transit_label, p_transit in TRANSITION_SCENARIOS.items():
            unemployment_model = WeibullUnemploymentModel(
                p_exit=p_transit, weibull_k=unemployment_k, weibull_lambda=unemployment_lambda
            )

            for salary in tqdm(SALARY_RANGE,
                               desc=f'salary [{market_scenario}/{transit_label}]'):
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
                                    'market_scenario':     market_scenario,
                                    'transition_scenario': transit_label,
                                    'p_transition':        p_transit,
                                    'program':   'pds'  if is_pds else 'iis3',
                                    'portfolio': portfolio_label,
                                    'salary':    salary,
                                    'age':       age,
                                    'sex':       sex,
                                    'sim_id':    i,
                                    **metrics,
                                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TEMP_DIR, 'h1_portfolio_freedom.csv'), index=False)
    print(f"Сохранено: {TEMP_DIR}/h1_portfolio_freedom.csv  ({len(df)} строк)")

    # ── описательная статистика по перцентилям (базовый сценарий) ────────────
    pctiles = [5, 25, 50, 75, 95]
    print("\n=== Описательная статистика (базовый рыночный + базовый трудовой сценарий) ===")
    base = df[(df['market_scenario'] == 'baseline') &
              (df['transition_scenario'] == 'baseline')]
    for metric in ['twr', 'irr', 'kz']:
        print(f"\n  {metric.upper()}:")
        print(f"  {'Портфель':15} " + " ".join(f"p{p:>3}" for p in pctiles) + "   mean    std")
        for pf in PORTFOLIOS:
            vals = base.loc[base['portfolio'] == pf, metric].dropna()
            if len(vals) == 0:
                continue
            pcts = np.percentile(vals, pctiles)
            print(f"  {pf:15} " + " ".join(f"{v:>6.1%}" for v in pcts)
                  + f"  {vals.mean():>6.1%}  {vals.std():>6.1%}")

    # ── описательная статистика по перцентилям ────────────────────────────────
    pctiles = [5, 25, 50, 75, 95]
    print("\n=== Описательная статистика (базовый сценарий безработицы) ===")
    base = df[df['transition_scenario'] == 'baseline']
    for metric in ['twr', 'irr', 'kz']:
        print(f"\n  {metric.upper()}:")
        print(f"  {'Портфель':15} " + " ".join(f"p{p:>3}" for p in pctiles) + "   mean    std")
        for pf in PORTFOLIOS:
            vals = base.loc[base['portfolio'] == pf, metric].dropna()
            pcts = np.percentile(vals, pctiles)
            print(f"  {pf:15} " + " ".join(f"{v:>6.1%}" for v in pcts)
                  + f"  {vals.mean():>6.1%}  {vals.std():>6.1%}")

