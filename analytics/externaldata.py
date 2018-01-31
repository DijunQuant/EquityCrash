from pandas_datareader.data import DataReader,get_quote_yahoo

import pandas as pd
from pandas_datareader.yahoo.quotes import _yahoo_codes
from pandas_datareader.google.quotes import _BaseReader
tickers=['aapl']

data_source='yahoo'

start_date='2016-01-01'
end_date='2017-01-10'

panel_data=DataReader(tickers,data_source,start_date,end_date)

_yahoo_codes.update({'Market Cap': 'j1'})

df = get_quote_yahoo('AAP')


import quandl
quandl.ApiConfig.api_key ='4GJtVewAnvmswbxrwGBu'

import static



alldata=[]
try:
    for stock in static.equitties_train:
        mydata = quandl.get_table('ZACKS/MKTV', ticker=stock)
        if len(mydata)>0:

            mydata['date']=pd.to_datetime(mydata['per_end_date'])
            mc = mydata.set_index('date')[['mkt_val']].sort_index().iloc[-1]['mkt_val']
            alldata.append((stock,round(mc)))
        else:
            alldata.append((stock,'nan'))
except:
    print(alldata)

#this is the good one
from pinance import Pinance

alldata=[]
try:
    for stock in static.equitties_train:
        share = Pinance(stock)
        try:
            share.get_quotes()
            mc=share.quotes_data['marketCap']
        except:
            alldata.append((stock, 'nan'))
            continue
        alldata.append((stock,round(mc/1000000)))
except:
    print(alldata)