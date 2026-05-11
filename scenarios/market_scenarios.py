"""
Рыночные сценарии для проверки устойчивости результатов
--------------------------------------------------------
Три сценария по параметрам активов:

  baseline   — параметры откалиброваны по историческим данным (MOEX 2012–2023)
  stress     — стрессовый: ниже доходность акций, выше волатильность,
               выше ставка/доходность облигаций, выше инфляция
  optimistic — оптимистичный: выше доходность акций, ниже волатильность,
               ниже ставка/доходность облигаций, ниже инфляция

Использование
  from scenarios.market_scenarios import MARKET_SCENARIOS, build_portfolio_for_scenario
  returns = build_portfolio_for_scenario(
      scenario='stress', indexes=indexes, structure=struct_dict,
      corr_matrix=CORR_4X4, n_years=15, n_simulations=300
  )
"""

import numpy as np
import pandas as pd
from models.securities.stocks import StockModel, StockParams
from models.securities.bonds import BondModel, BondParams
from models.securities.portfolio import PortfolioModel

# ── Множители для сдвига параметров относительно базового ─────────────────────
# Каждый сценарий задаётся как словарь множителей/сдвигов:
#   stock_mu_shift   — абсолютный сдвиг ожидаемой доходности акций (напр. -0.05 = -5 п.п.)
#   stock_sigma_mult — множитель волатильности акций
#   bond_b_shift     — абсолютный сдвиг долгосрочного среднего доходности облигаций
#   bond_sigma_mult  — множитель волатильности облигаций

MARKET_SCENARIOS: dict[str, dict] = {
    'baseline': {
        'label':            'Базовый (исторические параметры)',
        'stock_mu_shift':   0.00,
        'stock_sigma_mult': 1.00,
        'bond_b_shift':     0.00,
        'bond_sigma_mult':  1.00,
    },
    'stress': {
        'label':            'Стрессовый (↓доходность акций, ↑волатильность, ↑ставки)',
        'stock_mu_shift':  -0.05,   # доходность акций ниже на 5 п.п.
        'stock_sigma_mult': 1.40,   # волатильность акций выше на 40 %
        'bond_b_shift':     0.03,   # долгосрочная ставка облигаций выше на 3 п.п.
        'bond_sigma_mult':  1.30,   # волатильность облигаций выше на 30 %
    },
    'optimistic': {
        'label':            'Оптимистичный (↑доходность акций, ↓волатильность, ↓ставки)',
        'stock_mu_shift':   0.04,   # доходность акций выше на 4 п.п.
        'stock_sigma_mult': 0.75,   # волатильность акций ниже на 25 %
        'bond_b_shift':    -0.02,   # долгосрочная ставка облигаций ниже на 2 п.п.
        'bond_sigma_mult':  0.80,   # волатильность облигаций ниже на 20 %
    },
}


def build_portfolio_for_scenario(
    scenario: str,
    indexes: pd.DataFrame,
    structure: dict,
    corr_matrix: np.ndarray,
    n_years: int,
    n_simulations: int,
    yield_col: str = 'YIELD',
    a: float = 0.1,
    b: float = 0.07,
    dt: float = 1 / 252,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Строит портфель с параметрами, сдвинутыми согласно сценарию, и симулирует доходности.

    Возвращает матрицу (n_steps, n_simulations) — как PortfolioModel.simulate().
    """
    sc = MARKET_SCENARIOS[scenario]
    securities = []

    for asset_type, weight in structure.items():
        if weight == 0:
            continue

        if 'stock' in asset_type:
            mu_base   = indexes['R_t'].mean() * 252
            sigma_base = indexes['R_t'].std() * np.sqrt(252)
            params = StockParams(
                mu=mu_base + sc['stock_mu_shift'],
                sigma=sigma_base * sc['stock_sigma_mult'],
                weight=weight,
            )
            securities.append(StockModel(params))

        elif 'bond' in asset_type:
            col    = f'{yield_col}_{asset_type}'
            sigma_base = indexes[col].std()
            Y0     = float(indexes.loc[0, col])
            params = BondParams(
                sigma=sigma_base * sc['bond_sigma_mult'],
                b=b + sc['bond_b_shift'],
                a=a,
                Y0=Y0,
                weight=weight,
            )
            securities.append(BondModel(params))

    portfolio = PortfolioModel(assets=securities, corr_matrix=corr_matrix)
    return portfolio.simulate(
        n_years=n_years,
        n_simulations=n_simulations,
        dt=dt,
        show_progress=show_progress,
    )
