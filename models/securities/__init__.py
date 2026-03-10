import numpy as np
import pandas as pd

from scenarios import YIELD_COL, SECURITY_TYPES
from models.securities.bonds import BondModel, BondParams
from models.securities.stocks import StockModel, StockParams

from scenarios import YIELD_COL

a = 0.1
b = 0.07

def get_security_params(indexes: pd.DataFrame, a: float = a, b: float = b, structure: dict = None):

    securities = []
    SECURITY_TYPES = list(structure.keys())

    for security_type in SECURITY_TYPES:

        weight = structure[security_type]

        if 'stock' in security_type:

            mu = indexes['R_t'].mean() * 252
            sigma = indexes['R_t'].std() * np.sqrt(252)

            stock_params = StockParams(mu=mu, sigma=sigma, weight=weight)

            securities.append(StockModel(stock_params))

        elif 'bond' in security_type:

            sigma = indexes[f'{YIELD_COL}_{security_type}'].std()
            b = b
            a = a
            Y0 = indexes.loc[0, f'{YIELD_COL}_{security_type}']

            bond_params = BondParams(sigma=sigma, b=b, a=a, Y0=Y0, weight=weight)

            securities.append(BondModel(bond_params))

        else:
            raise ValueError(f'Invalid security type. Key word ("bond" or "stock") should be in security type param. Your input - {security_type}')

    return securities
