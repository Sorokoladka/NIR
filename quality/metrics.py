import numpy as np
from typing import Optional
from numpy_financial import irr as npf_irr


def savings_metric(final_accumulation: float) -> float:
    return final_accumulation

def roi_metric(final_accumulation: float, total_payments: float) -> float:
    if total_payments == 0:
        return np.nan
    return (final_accumulation - total_payments) / total_payments

def irr_metric(payments: np.ndarray, final_accumulation: float) -> float:
    n = len(payments) - 1  # payments[-1] == 0
    cf = np.concatenate([-payments[:n], [final_accumulation]])
    try:
        return float(npf_irr(cf))
    except:
        return np.nan

def twr_metric(portfolio_path: np.ndarray, payments: np.ndarray) -> float:
    n = len(payments) - 1
    returns = []
    # Шаг 0
    beginning_0 = 0 + payments[0]
    r0 = portfolio_path[0] / beginning_0 - 1 if beginning_0 != 0 else 0.0
    returns.append(r0)
    # Шаги 1..n-1
    for i in range(1, n):
        beginning = portfolio_path[i-1] + payments[i]
        r = portfolio_path[i] / beginning - 1 if beginning != 0 else 0.0
        returns.append(r)
    prod = np.prod([1 + r for r in returns])
    return prod ** (1 / len(returns)) - 1 if returns else np.nan

def kz_metric(pension: float, annual_salaries: np.ndarray, n: int) -> float:
    avg_final_monthly_salary = (annual_salaries[-(n+1):-1] / 12).mean()
    return pension / avg_final_monthly_salary if avg_final_monthly_salary != 0 else np.nan

def pension_metric(final_accumulation: float, annuity_months: Optional[float]) -> Optional[float]:
    if annuity_months is None:
        return None
    return final_accumulation / annuity_months