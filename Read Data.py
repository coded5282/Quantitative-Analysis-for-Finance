import datetime as dt
import matplotlib.pyplot as plt
# make graphs look better
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd # data analysis library
import pandas_datareader.data as web

style.use('ggplot') # one of the styles we can use

#==============================================================================
# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016,12,31)
# 
# # think of df as a spreadsheet
# df = web.DataReader('NASDAQ:TSLA', 'google', start, end)
# # default prints first 5 heads of data frame
# print(df.head())
# 
# df.to_csv('tsla.csv')
#==============================================================================

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

# get 10-day open high low close
df_ohlc = df['Close'].resample('10D').ohlc()
# get total volume over 10 days
df_volume = df['Volume'].resample('10D').sum()

# adds a separate index column with integers
df_ohlc.reset_index(inplace=True)

# convert to mdates format for matplotlib
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

#print(df.head())
#==============================================================================
# print(df[['Open','High']].head())
# 
# df['Close'].plot()
# plt.show()
#==============================================================================

# adding column of 100-day moving averages
#df['100ma'] = df['Close'].rolling(window=100, min_periods=0).mean()
#df.dropna(inplace=True) # removes entire rows with NaN
#print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

# making candlestick graphs
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
# dates is x-axis and volume is y-axis
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

#==============================================================================
# ax1.plot(df.index, df['Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])
#==============================================================================

plt.show()