import pandas_datareader as web
# import pandas as pd
# from pandas.tseries.offsets import BDay
# import holidays as hl
# import datetime
# import numpy as np
# import dateutil.relativedelta as rd
# import os


def get_prices(ticker, source, startDate, endDate, *args):

    if source == 'yahoo':
        try:
            result = web.DataReader(ticker, 'yahoo', startDate, endDate,)
            # TODO: add multiindex on columns if multiple columns chosen
            if len(args) > 0:
                result = result[[x for x in args]]
                result.columns = ticker
        except KeyError:
            print('may be wrong ticker name')
            result = None
        except Exception:
            print(Exception)
            result = None

    # TODO: stooq not extracting us index data. Only stocks
    if source == 'stooq':
        result = web.DataReader(ticker, source, startDate, endDate)
        result = result[(result.index > startDate) & (result.index < endDate)]
        result = result.iloc[::-1]

    if source == 'moex':
        result = web.DataReader(ticker, 'moex', startDate, endDate)
        result = result[result['BOARDID'] == 'TQBR']
        result = result[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
        result.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    if source == 'quandl':
        result = web.DataReader(ticker, 'quandl', startDate, endDate)
        result = result[['Open', 'High', 'Low', 'Close', 'Volume']]

    return result


# TODO: incorporate code fragment into function
# def get_data(ticker, source, start_t, end_t, market):
#     if market == 'RU':
#         ticker = ticker+'.ME'
#     if source == 'yahoo':
#         OHLC = data.DataReader(ticker, 'yahoo', start_t, end_t)
#     else:
#         files = os.listdir(source)
#         file = [x for x in files if ticker in x][0]
#         OHLC = pd.read_csv(os.path.join(source, file), delimiter=';', index_col=0, header=None)  # .iloc[:, :4]
#         OHLC.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap']
#         OHLC.index = pd.to_datetime(OHLC.index)
#         OHLC = OHLC[(OHLC.index <= end_t) & (OHLC.index > start_t)]
#         OHLC = OHLC.fillna(method='ffill')
#         OHLC = OHLC.iloc[::-1]
#
#     return OHLC
# def get_close_dataset(tickers, source, start_t, end_t, market, fields=['Close']):
#     # portfolio_components = dict()
#     closes = pd.DataFrame()
#     for ticker in tickers:
#         # closes[ticker] = get_data(ticker, source, start_t, end_t)['Close']
#         newcol = get_data(ticker, source, start_t, end_t, market)[fields]
#         newcol.columns = [x+'_'+ticker for x in fields]
#         closes = closes.join(newcol, how='outer')
#
#     closes.astype(float)
#     return closes
# startDate = '2018-1-1'
# endDate = '2020-5-11'
# ticker = 'MSFT.US'
# source = 'stooq'
# get_prices(ticker, source, startDate, endDate)