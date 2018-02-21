import analytics.dataprocess as dataprocess
import analytics.modeltrain as modeltrain
import analytics.strategy as strategy
import static
import os
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

pd.set_option('display.float_format', lambda x: '%.4f' % x)
import matplotlib.pyplot as plt
featurecolsbase=['alltradeimb_auto','trfimb_auto','trfcntimb_auto','totimb_auto','mindepthNYSE', 'mindepthNASDAQ', 'mindepthBATS','spread','litvolume']

resample_freq=1 #minute
df=pd.DataFrame()

equitydict=dict(static.equities_train)

equitylist=[x[0] for x in static.equities_train if ((x[0] in static.equities_done) and (equitydict[x[0]]>static.median_cap_threshold))]

#equitylist=[x[0] for x in static.equities_train if ((x[0] in static.equities_done) and (equitydict[x[0]]<static.median_cap_threshold))]
print(len(static.equities_done),len(equitylist))

for equity in [x for x in equitylist]:
    thisDF=dataprocess.loadFeatureDataBulk(equity,featurecolsbase,temperAtOpen=[],resample=resample_freq,enddate=pd.datetime(2017,8,2))
    if len(thisDF)>0:
        thisDF['stock']=equity
        df=pd.concat([df,thisDF])


######crash method 1
crash_thrshld=0.0175
#crash_thrshld=0.02
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxchg_abs']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)
df = df[df['firsttime']>pd.to_timedelta('3m')]

df['timebucket']=df.index.hour+np.round(df.index.minute/15)*0.25
print(df.groupby('timebucket')['target'].sum())


targetsummary=df.groupby(['stock','date'])['target'].max()
print(targetsummary.value_counts())

featurecols = featurecolsbase+[feature+'_1' for feature in featurecolsbase] +[feature+'_2' for feature in featurecolsbase]


kfoldds=0.1
kfoldds=0.05

dsratio=0.2
dsratio_day=0.1
dsratio_morning=0.3
target_prob=None

#dsratio_day=dsratio
#dsratio_morning=dsratio
#target_prob=99

seed=1


morningmodel=XGBClassifier(max_depth=3,subsample=0.5)
daymodel=XGBClassifier(max_depth=3,subsample=0.5)

#rewrite fit model with all data

day_threshold=modeltrain.modelfit(df[df['firsttime'] >= pd.to_timedelta('40m')], featurecols, daymodel, kfoldds, dsratio_day, seed, weightadj=False,target_prob=target_prob)
morning_threshold = modeltrain.modelfit(df[df['firsttime'] < pd.to_timedelta('40m')], featurecols, morningmodel, kfoldds, dsratio_morning, seed, weightadj=False,target_prob=target_prob)
print(morning_threshold,day_threshold)


#from sklearn.externals import joblib

#saved_morningmodel = joblib.load('model/small_morningmodel.pkl')
#saved_daymodel = joblib.load('model/small_daymodel.pkl')
#saved_morning_threshold = 0.77871634006500245
#saved_day_threshold=0.14158028960228117

paramset=[]
for revertwindow in [3,5]:
    for reverthreshold in [0,0.002]:
        for crashwindow in [30]:
            #for crashthreshold in [0.015]:
            for crashthreshold in [0.0175]:
                for usepredict in [True,False]:
                    for stoploss in [0.0175]:#was 0.002
                        for stopgain in [0.05]:
                            for timewindow in [10,30,60]:#minute
                                paramset.append({'revertwindow':revertwindow,
                                                 'reverthreshold':reverthreshold,
                                                 'crashwindow':crashwindow,
                                                 'crashthreshold':crashthreshold,
                                                 'usepredict':usepredict,
                                                 'stopgain':stopgain,
                                                 'stoploss':stoploss,
                                                 'timewindow':timewindow})

#pnlcols=['pnl_' + str(window) for window in timewindows]
#lengthcols=['length_' + str(window) for window in timewindows]

alltrades = strategy.backtest(equitylist,paramset,featurecolsbase,featurecols,(morningmodel,morning_threshold),(daymodel,day_threshold),pd.datetime(2017, 8, 3))


#alltrades_bysaved = strategy.backtest(equitylist,paramset,featurecolsbase,featurecols,timewindows,(saved_morningmodel,saved_morning_threshold),(saved_daymodel,saved_day_threshold),pd.datetime(2017, 8, 3))

def truncateFirstPNL(df):
    pos=df[df['pnl']>0]
    if len(pos)>0:
        return df[df['time']<=(pos.iloc[0]['time'])].reset_index(drop=True)[['pnl','length','time']]
    else:
        return df.reset_index(drop=True)[['pnl','length','time']]
reportDict={}
for i in range(len(paramset)):
#for i in [5]:
    trade = pd.DataFrame(alltrades[i], columns=['pnl','length','time', 'stock','date'])
    trade['date']=pd.to_datetime(trade['date'])
    print(i,paramset[i])
    #trade=trade[trade['length']<0]
    #test only do one trade a day
    trade = trade.groupby(['date','stock'])[['pnl','length','time']].first().reset_index()
    #test only do
    #trade = trade.groupby(['date','stock']).apply(lambda x: truncateFirstPNL(x)).reset_index() #good idea for small cap
    #trade.to_csv('Data/strategy/smallcap_'+str(i)+'.csv')
    dailypnl = pd.DataFrame({'count':trade.groupby('date')['stock'].count()})
    dailypnl['pnl']=trade.groupby('date')['pnl'].sum()
    reportDict[(paramset[i]['reverthreshold'],paramset[i]['revertwindow'],paramset[i]['usepredict'],paramset[i]['timewindow'])]=\
        (len(dailypnl),len(trade),dailypnl['pnl'].mean()/dailypnl['pnl'].std()*np.sqrt(250*len(dailypnl)/80))
    print(len(dailypnl),len(trade),dailypnl['pnl'].mean()/dailypnl['pnl'].std()*np.sqrt(250*len(dailypnl)/80))

for key in reportDict.keys():
    if key[2]==True:print(key,reportDict[key])
for key in reportDict.keys():
    if key[2]==False:print(key,reportDict[key])


for i in [19]:
    #if not paramset[i]['usepredict']:continue
    trade = pd.DataFrame(alltrades[i], columns=['pnl','length','time', 'stock','date'])
    #trade=trade[trade['length']<0]

    trade['date'] = pd.to_datetime(trade['date'])
    print(paramset[i])

    dailypnl = pd.DataFrame({'count': trade.groupby('date')['stock'].count()})
    dailypnl['pnl'] = trade.groupby('date')['pnl'].sum()
    # print(dailypnl[pnlcols].describe())
    #print(len(dailypnl), len(trade))
    dailypnl['pnl'].cumsum().plot()

