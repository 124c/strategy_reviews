import os
os.chdir('/Users/mmajidov/Projects/platform')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datareader import pandas_reader as dr

def zscore_std(series):
    return ((series - series.mean()) / np.std(series)).rename("z-score")

tickers = ['MA', 'V']
dataset = dr.get_prices(tickers, 'yahoo', '2004-12-31', '2013-12-31', 'Adj Close').dropna()
data_year = dataset.loc['2008':'2013-11-01']
dataset.loc['2012':'2013-11-01'].plot()
data_year.plot()
x_train = sm.add_constant(data_year[tickers[1]])
model = sm.OLS(data_year[tickers[0]], x_train)
result = model.fit()
hedge_ratio = result.params[1]
hedge_ratio
spread = data_year[tickers[0]] - data_year[tickers[1]] * hedge_ratio
spread_zscore = zscore_std(spread)
# spread_zscore.hist()
# pd.DataFrame({'ITUB4.SA': data_year[tickers[0]],
#               'Ccro3.SA * hedge': data_year[tickers[1]] * hedge_ratio}).plot()