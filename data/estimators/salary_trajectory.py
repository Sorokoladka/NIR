if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    dfs = []
    for year in range(2007, 2023, 2):

        try:
            path = f'/Users/aasorokolad/PycharmProjects/NIR/data/demogr_salaries/{year}.xls'
            data_temp = pd.read_excel(path)
        except:
            path = f'/Users/aasorokolad/PycharmProjects/NIR/data/demogr_salaries/{year}.xlsx'
            data_temp = pd.read_excel(path)

        if year == 2007:
            columns = np.append(data_temp.iloc[4],'year')

        data_temp.rename(columns={data_temp.columns[0]: 'fictious'}, inplace=True)
        data_temp['year'] = year

        dfs.append(pd.DataFrame(data_temp[data_temp['fictious'].apply(lambda x: str(x).replace(' ','')) == 'Все работники'.replace(' ','')]))


    data = pd.concat(dfs)
    data.columns = columns
    data.drop(columns=np.nan, inplace=True)

    # data.to_csv('/Users/aasorokolad/PycharmProjects/NIR/data/demogr_salaries/demogr_salaries_agg.csv', index=False)

