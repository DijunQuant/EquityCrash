import numpy as np
import pandas as pd
import matplotlib as plt
import analytics.dataprocess as dataprocess



def analyze(df,bin_sec):
    df['change']=df['mid'].diff()
    df['trftradeimb']=df['tradeimb']+df['trfimb']
    df['othertradeimb']=df['tradeimb']+df['otherimb']
    df['alltradeimb']=df['tradeimb']+df['otherimb']+df['trfimb']
    newdf = df.groupby(pd.TimeGrouper(freq=str(bin_sec)+'S'))\
        ['change','trfimb','tradeimb','otherimb','trftradeimb','othertradeimb','alltradeimb','totimb'].sum()
    return newdf


############sandbox###############


equity='ZTS'
datestr='2017-11-02'

rawdata=dataprocess.getData(equity,datestr)

#df['tradeimbNew_cum']=df['tradeimb_cum']+df['otherimb_cum']
print(rawdata[['buy','sell','buyother','sellother','unknownother','buytrf','selltrf','unknowntrf','buytot','selltot']].sum())

newdf=analyze(rawdata,30)
print(np.round(newdf[['change','trfimb','tradeimb','otherimb','totimb']].corr(),2))

autocorrDF = analyze(rawdata,10)[['othertradeimb','totimb','trfimb']].rolling(60).apply(lambda x: pd.Series(x).autocorr(1))
autocorrDF=autocorrDF.merge(rawdata[['mid']],left_index=True,right_index=True,how='left')

autocorrDF.plot(secondary_y='mid').get_figure().savefig(equity+'-'+datestr+'.autocorr.png')


rawdata[['mid','tradeimb_cum','totimb_cum','otherimb_cum','trfimb_cum']].plot(secondary_y=['mid']).\
    get_figure().savefig(equity+'-'+datestr+'.imb.png')

rawdata[['all_bid','all_ask']].rolling(60).mean().merge(rawdata[['mid']],left_index=True,right_index=True).plot(secondary_y='mid')

rawdata[['all_bid','all_ask']].rank().rolling(600).mean().merge(rawdata[['mid']],left_index=True,right_index=True).plot(secondary_y='mid')

rawdata[['tradeimb','totimb','otherimb','trfimb']].rolling(600).mean().merge(rawdata[['mid']],left_index=True,right_index=True).plot(secondary_y='mid')


rawdata[['all_bid','all_ask']].diff().rolling(600).sum().merge(rawdata[['mid']],left_index=True,right_index=True).plot(secondary_y='mid')
################other diagnosis#################

datestr='2017-09-12'

for equity in ['ABBV','ABT','ACN','ATVI','MMM']:
    df=dataprocess.getData(equity,datestr)
    newdf=analyze(df,30)
    print(equity)
    print(np.round(newdf[['change','trfimb','tradeimb','otherimb','totimb']].corr(),2))


df.dropna()[['vwapspread_trf']].hist(weights=df.dropna()['trf'].values,bins=50)

df.dropna()[['vwapspread_trf']].hist(bins=50)

df['vwap_buy']=df['buyamt']/df['buy']
df['vwap_sell']=df['sellamt']/df['sell']
df['vwapspread_buy']=df['vwap_buy']-df['bidpx'].shift()
df['vwapspread_sell']=df['vwap_sell']-df['askpx'].shift()
df[['vwapspread_buy','vwapspread_sell']].describe()
