"""
Гипотезы H2 и H3
----------------
H2: Существует пороговое значение взноса в ПДС, при котором предельная выгода
    от софинансирования уравнивается с предельными издержками (комиссии УК и фонда).
    При взносах выше этого порога дополнительная доходность ПДС снижается, создавая
    экономические стимулы для диверсификации в пользу ИИС-3.

H3: Механизм государственного софинансирования создаёт условия для убывающей
    доходности инвестора при увеличении взносов.

Дизайн эксперимента
-------------------
Переменная развёртки: payment_rate ∈ [0.5 %, 24 %], шаг 0.5 % → 47 значений
Контроль: salary (покрывает все 3 тира со-фин.) × n_simulations

Ключевая идея идентификации
  - Для H2/H3 необходимо изолировать эффект со-финансирования от эффекта портфеля.
    Поэтому ПДС и ИИС-3 используют ОДИНАКОВЫЕ реализации доходностей (один портфель).
  - Разница метрик ПДС − ИИС-3 = чистый эффект (со-финансирование − комиссии).

Теоретические пороги (аналитически)
  salary < 80 k:      threshold_rate = 36 000 / (salary × 12)
  80 k ≤ salary < 150 k: threshold_rate = 72 000 / (salary × 12)
  salary ≥ 150 k:     threshold_rate = 144 000 / (salary × 12)

Численные точки перегиба
  По итогам симуляции для каждого salary считается:
    advantage(r) = E[ROI_ПДС(r)] − E[ROI_ИИС-3(r)]
  Точка перегиба H2: argmax_r advantage(r)   — при этом rate ПДС перестаёт расти
  Точка убывания H3: первый r, при котором d(E[ROI_ПДС])/dr < 0
  Сохраняются в отдельный CSV.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import argrelextrema

from models.securities.portfolio import PortfolioModel
from models.programs.base import ProgramInput
from models.programs.pds import PDSProgram
from models.programs.iis3 import IIS3Program
from models.macro.salary import StochasticSalaryModel
from models.macro.unemployment import WeibullUnemploymentModel

from scenarios import (indexes, structure, life_table, YIELD_COL,
                       unemployment_k, unemployment_p, unemployment_lambda)
from models.securities import get_security_params

# ── параметры ─────────────────────────────────────────────────────────────────
N_SIMULATIONS = 300
N_YEARS       = 15
TAX_RATE      = 0.13
AGE           = 40
SEX           = 'M'

PAYMENT_RATES = np.round(np.arange(0.5, 24.5, 0.5) / 100, 4)   # 48 значений
SALARY_RANGE  = [50_000, 100_000, 150_000, 200_000]

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

# ИИС-3: сбалансированный 50/50 с расщеплением облигаций как у НПФ
IIS3_STRUCT = {
    'stock':     0.50,
    'gov_bond':  0.50 * BOND_RATIOS['gov_bond'],
    'corp_bond': 0.50 * BOND_RATIOS['corp_bond'],
    'mun_bond':  0.50 * BOND_RATIOS['mun_bond'],
}
PDS_STRUCT = structure.iloc[-1][ASSET_ORDER].to_dict()


def analytical_threshold(salary: float) -> float:
    """Аналитический порог взноса (как доля от зарплаты) по формуле со-финансирования."""
    annual_salary = salary * 12
    if salary < 80_000:
        return 36_000 / annual_salary
    elif salary < 150_000:
        return 72_000 / annual_salary
    else:
        return 144_000 / annual_salary


def find_inflection_points(rates: np.ndarray, advantage: np.ndarray) -> dict:
    """
    Находит точки перегиба кривой advantage(rate):
      - peak: argmax advantage (H2 — точка, где ПДС перестаёт выигрывать)
      - zero_cross: первый rate, где advantage < 0 (ИИС-3 начинает обгонять)
      - pds_peak: argmax E[ROI_PDS] (H3 — убывающая доходность внутри ПДС)
    Возвращает dict с численными значениями rate.
    """
    result = {
        'advantage_peak_rate':   float(rates[np.argmax(advantage)]),
        'advantage_peak_value':  float(np.max(advantage)),
        'zero_cross_rate':       None,
    }
    for i in range(len(advantage) - 1):
        if advantage[i] >= 0 and advantage[i + 1] < 0:
            result['zero_cross_rate'] = float(rates[i])
            break
    return result


if __name__ == '__main__':

    # ── симулируем ОДИН общий портфель для изоляции эффекта со-финансирования ──
    # ПДС запускается с PDS_STRUCT, ИИС-3 — тоже с PDS_STRUCT (один набор доходностей)
    print("Симулирую общий базовый портфель...")
    securities_pds  = get_security_params(indexes=indexes, structure=PDS_STRUCT)
    portfolio_base  = PortfolioModel(assets=securities_pds, corr_matrix=CORR_4X4)
    base_returns    = portfolio_base.simulate(
        n_years=N_YEARS, n_simulations=N_SIMULATIONS, dt=1/252, show_progress=False
    )

    unemployment_model = WeibullUnemploymentModel(
        p_exit=unemployment_p, weibull_k=unemployment_k, weibull_lambda=unemployment_lambda
    )
    salary_model = StochasticSalaryModel(initial_age=AGE)

    # ── основной цикл: salary × payment_rate × program ────────────────────────
    rows = []
    for salary in tqdm(SALARY_RANGE, desc='salary'):
        for payment_rate in PAYMENT_RATES:
            for i in range(N_SIMULATIONS):
                rates_vec = list(base_returns[:, i][::252])
                n = len(rates_vec)

                base_params = dict(
                    n=n, age=AGE, sex=SEX,
                    rates=rates_vec,
                    payment_mode='relative',
                    payment_rate=float(payment_rate),
                    initial_salary=salary,
                    tax_deduction_rate=TAX_RATE,
                    salary_model=salary_model,
                    unemployment_model=unemployment_model,
                )

                # ПДС
                pds_prog = PDSProgram(
                    params=ProgramInput(**base_params), life_table=life_table
                )
                pds_prog.run()
                m_pds = pds_prog.compute_metrics()

                # ИИС-3 (те же реализации доходностей!)
                iis_prog = IIS3Program(
                    params=ProgramInput(**base_params), life_table=life_table
                )
                iis_prog.run()
                m_iis = iis_prog.compute_metrics()

                rows.append({
                    'salary':       salary,
                    'payment_rate': float(payment_rate),
                    'sim_id':       i,
                    'roi_pds':      m_pds['roi'],
                    'irr_pds':      m_pds['irr'],
                    'twr_pds':      m_pds['twr'],
                    'savings_pds':  m_pds['savings'],
                    'kz_pds':       m_pds['kz'],
                    'roi_iis':      m_iis['roi'],
                    'irr_iis':      m_iis['irr'],
                    'twr_iis':      m_iis['twr'],
                    'savings_iis':  m_iis['savings'],
                    'kz_iis':       m_iis['kz'],
                })

    df = pd.DataFrame(rows)
    df.to_csv('temp_data/h2h3_raw.csv', index=False)
    print(f"Сырые данные: temp_data/h2h3_raw.csv  ({len(df)} строк)")

    # ── постобработка: численные точки перегиба ───────────────────────────────
    summary_rows = []
    for salary in SALARY_RANGE:
        sub = df[df['salary'] == salary]
        # средние по симуляциям для каждого payment_rate
        agg = sub.groupby('payment_rate')[['roi_pds', 'roi_iis', 'irr_pds', 'irr_iis']].mean()
        agg = agg.reset_index().sort_values('payment_rate')

        rates_arr     = agg['payment_rate'].values
        advantage_roi = (agg['roi_pds'] - agg['roi_iis']).values    # H2
        roi_pds_arr   = agg['roi_pds'].values                        # H3

        inflection = find_inflection_points(rates_arr, advantage_roi)

        # H3: первый rate после которого d(ROI_ПДС)/d(rate) < 0
        d_roi_pds = np.diff(roi_pds_arr)
        h3_rate = None
        for j, d in enumerate(d_roi_pds):
            if d < 0:
                h3_rate = float(rates_arr[j])
                break

        theo_thresh = analytical_threshold(salary)

        summary_rows.append({
            'salary':                        salary,
            'analytical_threshold_rate':     round(theo_thresh, 4),
            'h2_advantage_peak_rate':        round(inflection['advantage_peak_rate'], 4),
            'h2_advantage_peak_value':       round(inflection['advantage_peak_value'], 4),
            'h2_zero_cross_rate':            inflection['zero_cross_rate'],
            'h3_diminishing_return_rate':    h3_rate,
            'max_roi_pds':                   round(float(roi_pds_arr.max()), 4),
            'min_roi_pds':                   round(float(roi_pds_arr.min()), 4),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv('temp_data/h2h3_inflection_points.csv', index=False)
    print("\nТочки перегиба:")
    print(summary.to_string(index=False))
    print(f"\nСохранено: temp_data/h2h3_inflection_points.csv")
