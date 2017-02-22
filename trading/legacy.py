import datetime
import pandas as pd
import pandas_datareader as web
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
#sp500 = data.get_data_yahoo('%EGSPC',start=datetime.datetime(2014,01,01), end=datetime.datetime(2014,04,01))
sp500 = web.DataReader('IDEA.NS', "yahoo", datetime.datetime(2016,1,1), datetime.datetime(2017,1,30))
print sp500.tail()

sp500['H-L']=sp500[sp500['High']-sp500['Low']]
close=sp500['Adj Close']
ma1 = pd.rolling_mean(close,50)
ma2 = pd.rolling_mean(close,20)

ax1 = plt.subplot(2,1,1)
ax1.plot(sp500['Adj Close'],label='IDEA')
ax1.plot(ma1,label='50MA')
plt.legend()

ax2 = plt.subplot(2,1,2,sharex = ax1)
ax2.plot(sp500['H-L'],label='H-L')
plt.legend()

plt.show()
