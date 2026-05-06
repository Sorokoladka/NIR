"""
Гипотеза H4
-----------
Результативность программ неоднородна по группам населения: демографические и
социальные признаки индивида (возраст, пол, доход) определяют более выгодную
для него программу.

Дизайн эксперимента
-------------------
Сетка: age_group × sex × программа (ПДС + ИИС-3 ×3 аллокации)
Представительная зарплата: из demogr_salaries_agg.csv (данные 2021 года)
Метрика сравнения: KZ (коэффициент замещения) — основной показатель пенсионной
  адекватности; дополнительно: first_pension, IRR, ROI
Победитель per ячейки: argmax(E[KZ]) по всем программам

Оценка численности потенциальных вкладчиков
-------------------------------------------
Источник: Обследование рабочей силы Росстат 2024 (ZAN 2024_сайт.sav)
Фильтр: наёмные работники (ST_ZAN ∈ {41, 42, 43, 44}), возраст 18–60
Нормировка: к официальным данным Росстата о занятости
  наёмные работники 18–60 ≈ 56.5 млн (2024)
Результат: для каждой ячейки — численность населения в млн чел.,
  для которых оптимальна та или иная программа
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import pyreadstat

from models.securities.portfolio import PortfolioModel
from models.programs.base import ProgramInput
from models.programs.pds import PDSProgram
from models.programs.iis3 import IIS3Program
from models.macro.salary import StochasticSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel

from scenarios import (indexes, structure, life_table, YIELD_COL,
                       unemployment_k, unemployment_p, unemployment_lambda)
from models.securities import get_security_params
import os

# ── параметры ─────────────────────────────────────────────────────────────────
N_SIMULATIONS = 300
N_YEARS       = 15
PAYMENT_RATE  = 0.06
TAX_RATE      = 0.13

ASSET_ORDER = ['stock', 'gov_bond', 'corp_bond', 'mun_bond']
YIELD_COLS  = [f'{YIELD_COL}_{a}' for a in ASSET_ORDER]
CORR_4X4    = indexes[YIELD_COLS].corr().values

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

# ── демографическая сетка (из demogr_salaries_agg.csv, данные 2021) ───────────
# Представительный возраст = середина 5-летнего интервала
AGE_GROUPS = {
    '20-24': {'rep_age': 22, 'salary_col': 'от 20 до 24 лет'},
    '25-29': {'rep_age': 27, 'salary_col': 'от 25 до 29 лет'},
    '30-34': {'rep_age': 32, 'salary_col': 'от 30 до 34 лет'},
    '35-39': {'rep_age': 37, 'salary_col': 'от 35 до 39 лет'},
    '40-44': {'rep_age': 42, 'salary_col': 'от 40 до 44 лет'},
    '45-49': {'rep_age': 47, 'salary_col': 'от 45 до 49 лет'},
    '50-54': {'rep_age': 52, 'salary_col': 'от 50 до 54 лет'},
    '55-59': {'rep_age': 57, 'salary_col': 'от 55 до 59 лет '},  # пробел в ключе как в CSV
}
SEX_RANGE = ['M', 'F']

# ── загрузка представительных зарплат ────────────────────────────────────────
def load_representative_salaries() -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, 'data', 'demogr_salaries', 'demogr_salaries_agg.csv')
    sal_agg = pd.read_csv(path)
    row_2021 = sal_agg[sal_agg['year'] == 2021].iloc[0]
    salaries = {}
    for group, info in AGE_GROUPS.items():
        salaries[group] = float(row_2021[info['salary_col']])
    return salaries


# ── загрузка весов ЗАН для оценки численности вкладчиков ─────────────────────
def load_zan_weights() -> pd.DataFrame:
    """
    Возвращает DataFrame с популяционными весами (млн чел.) по (age_group, sex).
    Нормировка: наёмные работники 18–60 ≈ 56.5 млн (Росстат, 2024).
    """
    TOTAL_EMPLOYED = 56.5e6   # официальная оценка Росстат

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, 'data', 'ZAN 2024_сайт.sav')
    df, _ = pyreadstat.read_sav(path)

    employed = df[df['ST_ZAN'].isin([41.0, 42.0, 43.0, 44.0])]
    wa = employed[(employed['NAS_VOZR'] >= 18) & (employed['NAS_VOZR'] <= 60)].copy()

    age_bins   = [18, 25, 30, 35, 40, 45, 50, 55, 60]
    age_labels = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
    wa['age_group'] = pd.cut(wa['NAS_VOZR'], bins=age_bins, labels=age_labels)
    wa['sex']       = wa['NAS_POL'].map({1.0: 'M', 2.0: 'F'})

    vesa_total  = wa['VESA'].sum()
    scale       = TOTAL_EMPLOYED / vesa_total     # приводим к реальной численности

    pop = (
        wa.groupby(['age_group', 'sex'], observed=True)['VESA']
        .sum()
        .reset_index()
    )
    pop['population_mln'] = pop['VESA'] * scale / 1e6
    pop = pop.drop(columns='VESA')
    return pop


if __name__ == '__main__':

    print("Загружаю данные...")
    rep_salaries = load_representative_salaries()
    zan_weights  = load_zan_weights()

    print("Симулирую портфели...")
    simulated_returns = {}
    for label, struct_dict in PORTFOLIOS.items():
        securities = get_security_params(indexes=indexes, structure=struct_dict)
        portfolio  = PortfolioModel(assets=securities, corr_matrix=CORR_4X4)
        simulated_returns[label] = portfolio.simulate(
            n_years=N_YEARS, n_simulations=N_SIMULATIONS, dt=1/252, show_progress=False
        )

    unemployment_model = WeibullUnemploymentModel(
        p_exit=unemployment_p, weibull_k=unemployment_k, weibull_lambda=unemployment_lambda
    )

    # ── основной цикл ─────────────────────────────────────────────────────────
    rows = []
    for age_group, info in tqdm(AGE_GROUPS.items(), desc='age_group'):
        age    = info['rep_age']
        salary = rep_salaries[age_group]
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
                    m = prog.compute_metrics()

                    rows.append({
                        'age_group':   age_group,
                        'rep_age':     age,
                        'sex':         sex,
                        'salary':      round(salary),
                        'program':     'pds' if is_pds else 'iis3',
                        'portfolio':   portfolio_label,
                        'sim_id':      i,
                        'kz':          m['kz'],
                        'first_pension': prog.first_pension,
                        'irr':         m['irr'],
                        'roi':         m['roi'],
                        'savings':     m['savings'],
                    })

    df = pd.DataFrame(rows)
    df.to_csv('temp_data/h4_raw.csv', index=False)
    print(f"Сырые данные: temp_data/h4_raw.csv  ({len(df)} строк)")

    # ── карта победителей ─────────────────────────────────────────────────────
    # Для каждой ячейки (age_group, sex): находим программу с max E[KZ]
    mean_kz = (
        df.groupby(['age_group', 'sex', 'portfolio'])['kz']
        .mean()
        .reset_index()
        .rename(columns={'kz': 'mean_kz'})
    )

    # Лучшее среди всех портфелей для каждого типа программы
    # iis3_best = max по трём ИИС-3 аллокациям
    iis3_kz = mean_kz[mean_kz['portfolio'].str.startswith('iis3')]
    iis3_best = (
        iis3_kz.groupby(['age_group', 'sex'])['mean_kz']
        .max()
        .reset_index()
        .rename(columns={'mean_kz': 'kz_iis3_best'})
    )
    pds_kz = (
        mean_kz[mean_kz['portfolio'] == 'pds_avg'][['age_group', 'sex', 'mean_kz']]
        .rename(columns={'mean_kz': 'kz_pds'})
    )

    winner_map = iis3_best.merge(pds_kz, on=['age_group', 'sex'])
    winner_map['winner']   = np.where(winner_map['kz_pds'] >= winner_map['kz_iis3_best'],
                                       'pds', 'iis3')
    winner_map['kz_delta'] = (winner_map['kz_pds'] - winner_map['kz_iis3_best']).round(4)

    # Присоединяем популяционные веса
    winner_map = winner_map.merge(
        zan_weights.assign(age_group=zan_weights['age_group'].astype(str)),
        on=['age_group', 'sex'],
        how='left'
    )

    winner_map.to_csv('temp_data/h4_winner_map.csv', index=False)

    # ── сводка потенциальных вкладчиков ──────────────────────────────────────
    pop_summary = (
        winner_map.groupby('winner')['population_mln']
        .sum()
        .reset_index()
        .rename(columns={'population_mln': 'population_mln'})
    )
    pop_summary['share_pct'] = (
        pop_summary['population_mln'] / pop_summary['population_mln'].sum() * 100
    ).round(1)

    pop_summary.to_csv('temp_data/h4_population_summary.csv', index=False)

    print("\n=== Карта победителей (первые строки) ===")
    print(winner_map[['age_group', 'sex', 'kz_pds', 'kz_iis3_best',
                       'winner', 'kz_delta', 'population_mln']].to_string(index=False))
    print("\n=== Оценка численности потенциальных вкладчиков ===")
    print(pop_summary.to_string(index=False))
    print("\nСохранено: temp_data/h4_raw.csv, h4_winner_map.csv, h4_population_summary.csv")
