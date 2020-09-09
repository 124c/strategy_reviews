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

tickers = ['ITUB4.SA', 'Ccro3.SA']
tickers = ['Brap4.SA', 'Csna3.SA']
dataset = dr.get_prices(tickers, 'yahoo', '2004-12-31', '2013-12-31', 'Adj Close').dropna()
data_year = dataset.loc['2005':'2006-11-01']
# data_year.plot()
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

# 0 - no position, 1 - long, -1 - short
positions = [0]
number_pos = [0]
# n_pos = 0
for i in range(1, len(spread_zscore)):
    current_position = positions[-1]  # today I look at active position that was set yestraday for today
    positions.append(trading_rule(current_position, spread_zscore[i]))

signals = pd.DataFrame({'L':data_year[tickers[0]],
                        'S':data_year[tickers[1]]* hedge_ratio,
                        'Spread_zscore':spread_zscore,
                        'positions':positions},
                        index=spread.index)
signals['positions'] = signals['positions'].shift().fillna(0)
signals['position_numbers'] = get_position_numbers(signals['positions'].tolist())
signals = limit_position_time(signals, 50)  # simple limiter for positions longer than 50 days
# I look at the data at the end of the day!
# the trading decision comes in force next day!

C = 0.05
log_L = pd.Series(np.log(data_year[tickers[0]]))
log_S = pd.Series(np.log(data_year[tickers[1]]))
signals['Log_L'] = log_L.diff()
signals['Log_S'] = log_S.diff()
signals['log_returns'] = np.where(signals['positions'] == 1,
                                  signals['Log_L'] - signals['Log_S']*hedge_ratio,
                                  np.where(signals['positions'] == -1,
                                           -1*(signals['Log_L'] - signals['Log_S']*hedge_ratio),
                                           0))
signals['log_returns_costs'] = np.where(signals['positions'] != 0,
                                    signals['log_returns']+2*np.log((1-C)/(1+C)),0)
signals['simple_returns'] = np.exp(signals['log_returns'])-1
trades = get_cross_section(signals, 0.005)

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
fig.suptitle('Positions and signals')
axs[0].plot(signals['L'], color='orange', label='L', fillstyle='none')
axs[0].plot(signals['S'], color='blue', label='S+hedge', fillstyle='none')
axs[0].fill_between(signals.index,
                    signals[['L','S']].min().min(),
                    signals[['L','S']].max().max(), where=signals['positions']==1,
                    facecolor='green', alpha=0.5,)
axs[0].fill_between(signals.index,
                    signals[['L','S']].min().min(),
                    signals[['L','S']].max().max(), where=signals['positions']==-1,
                    facecolor='red', alpha=0.5,)
axs[0].legend()
axs[1].plot(signals['Spread_zscore'].index, signals['Spread_zscore'].values)
axs[1].axhline(-2, color='green', lw=2, alpha=0.5)
axs[1].axhline(-0.5, color='green', lw=2, alpha=0.5)
axs[1].axhline(2, color='red', lw=2, alpha=0.5)
axs[1].axhline(0.75, color='red', lw=2, alpha=0.5)
plt.show()