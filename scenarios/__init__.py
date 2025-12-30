import pandas as pd
import numpy as np

import json
from pathlib import Path
import pyreadstat
from scipy.stats import weibull_min

PROJECT_PATH = Path(__file__).parent
CFG_PATH = PROJECT_PATH.joinpath('cfg.json')

with open(CFG_PATH.as_posix(), 'r') as f:
    CFG = json.load(f)

INDEXES = CFG["INDEXES"]
SECURITY_TYPES = CFG["SECURITY_TYPES"]
DATE_COL = CFG["DATE_COL"]
YIELD_COL = CFG["YIELD_COL"]

parent_dir = 'data'
indexes_dir = 'indexes'

def process_data(df, security_type):
    df_cpy = df.copy()
    df_cpy[DATE_COL] = pd.to_datetime(df_cpy[DATE_COL], format='%d.%m.%Y')

    if 'bond' in security_type:

        df_cpy[YIELD_COL] = df_cpy[YIELD_COL].str.replace(',', '.').astype(float)
        df_cpy[YIELD_COL] = df_cpy[YIELD_COL].apply(lambda x: (x) / 100)
        df_cpy = df_cpy.dropna()

    elif 'stock' in security_type:

        df_cpy['CLOSE'] = df_cpy['CLOSE'].str.replace(',', '.').astype(float)
        df_cpy['R_t'] = df_cpy['CLOSE'] / df_cpy['CLOSE'].shift(1) - 1
        df_cpy[YIELD_COL] = (1 + df_cpy['R_t']).rolling(window=252).apply(lambda x: x.prod()) - 1

    else:
        raise ValueError(f'Invalid security type. Key word ("bond" or "stock") should be in security type param. Your input - {security_type}')

    df_cpy = df_cpy[(df_cpy[DATE_COL] >= '03.05.2012') & (df_cpy[DATE_COL] <= '31.05.2023')].reset_index(drop=True) #crutch
    return df_cpy

# структура портфелей
structure = pd.read_excel(f'{parent_dir}/structure.xlsx').set_index('id')

# индексы
for ind, security_type in zip(INDEXES, SECURITY_TYPES):
    load_data = pd.read_csv(f'{parent_dir}/{indexes_dir}/{ind}.csv', delimiter=';', encoding='cp1251')
    processed_data = process_data(df=load_data, security_type=security_type)

    if ind == INDEXES[0]:
        indexes = processed_data.copy()
        first_security_type = security_type
    else:
        indexes_cpy = indexes.copy()
        if ind == INDEXES[-1]:
            suffixes = (f'_{first_security_type}', f'_{security_type}')
        else:
            suffixes = ('', f'_{security_type}')

        indexes = indexes_cpy.merge(processed_data, on=DATE_COL, suffixes=suffixes)

# пожизненные выплаты
life_table = pd.read_csv(f'{parent_dir}/life_duration.csv', index_col='age')

# параметры для безработицы

df = pd.read_excel(f'{parent_dir}/unemployment.xlsx')['Численность выбывших работников в процентах к списочной численности работников\n']
unemployment_p = ((df/100 + 1).prod())**(1/len(df)) - 1

df, meta = pyreadstat.read_sav(f'{parent_dir}/ZAN 2024_сайт.sav')
unempl_data = df[(df['NAS_VOZR'] >= 18) & (df['BZ_PSK'] > 0)]['BZ_PSK'].to_list()
params = weibull_min.fit(unempl_data, floc=0)
unemployment_k, unemployment_lambda = params[0], params[2]