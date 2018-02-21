import numpy as np
import pandas as pd
import analytics.dataprocess as dataprocess

def getPosition(df,revertwindow,crashwindow,crashthreshold,revertthreshold,usepredict,stoploss,stopgain,timewindow,cutoffEOD=10):
    initial=df['mid'].values[0]
    df['diff']=df['mid'].diff()/initial
    df['max']=df['mid'].rolling(window=crashwindow).max()
    df['min']=df['mid'].rolling(window=crashwindow).min()
    df['range']=(df['max']-df['min'])/initial
    df['crashsign']=np.sign(df['mid'].diff(crashwindow))
    df['signed_chg']=df['mid'].diff(revertwindow)/initial*df['crashsign']
    #df['flag']=df.apply(lambda r: 1 if (r['range']>crashthreshold and r['signed_chg']<-revertsize) else 0, axis=1)
    df['cum_predict']=df['predict'].rolling(window=crashwindow).max()
    #flag has the sign of trade
    df['flag']=df.apply(lambda r: -1*r['crashsign'] if (r['range']>crashthreshold and r['signed_chg']<-revertthreshold) else 0,axis=1)
    cutofftime=df['firsttime'][-1]-pd.to_timedelta(str(cutoffEOD)+'m')
    result=[]
    if usepredict:
        tradeopportunity=df[(df['cum_predict']>0) & (df['flag']!=0) & (df['firsttime']>pd.to_timedelta('15m')) &(df['firsttime']<cutofftime)]
    else:
        tradeopportunity = df[(df['flag']!= 0) & (df['firsttime']>pd.to_timedelta('15m'))&(df['firsttime']<cutofftime)]
    while len(tradeopportunity)>0:
        df['pos'] = 0
        df['trdchg'] = 0
        df['trdtime'] = 0
        #print(tradeopportunity[['flag','crashsign','signed_chg','firsttime','cum_predict','range']].head())
        entry= df[tradeopportunity.index[0]:].index[1] #position start after seeing trigger
        #print(entry,df.loc[tradeopportunity.index[0],'flag'])
        df.loc[entry:,'pos']=df.loc[tradeopportunity.index[0],'flag']
        df.loc[entry:, 'trdchg'] = df.loc[entry:, 'diff'].cumsum()*df.loc[tradeopportunity.index[0],'flag']
        df.loc[entry:, 'trdtime'] = df.loc[entry:].index-entry
        df['lasttrdchg'] = df['trdchg'].shift()
        df['keep'] = df.apply(
                lambda r: 1 if (r['trdtime'] < pd.to_timedelta(str(timewindow) + 'm') and \
                                r['lasttrdchg'] > -stoploss and \
                                # r['lasttrdchg'] > -0.001 and \
                                r['lasttrdchg'] <= stopgain) else 0, axis=1)
        exit = df[(df['pos'] != 0) & (df['keep'] < 1)]
        if len(exit > 0):
            df.loc[exit.index[0]:, 'keep'] = 0
        else:
            exit=df.iloc[-1:]
        result.append([(df['pos'] * df['keep'] * df['diff']).sum(),(df['pos'] * df['keep']).sum(), entry])
        tradeopportunity=tradeopportunity[tradeopportunity.index>exit.index[0]+pd.to_timedelta('1m')]
    return result

def backtest(equitylist,paramset,featurecolsbase,featurecols,morningmodel,daymodel,startdate,enddate=pd.datetime(2017,11,29)):
    alltrades=[[] for key in paramset]
    # for stock in ['AAP']:
    for stock in equitylist:
        testDF = dataprocess.loadFeatureDataBulk(stock, featurecolsbase, temperAtOpen=[], truncPostMove=False,
                                                 startdate=startdate,enddate=enddate)
        if len(testDF)<1:
            print(stock,'load data fail')
            continue
        if morningmodel[1]==None:
            testDF['predict_m'] = morningmodel[0].predict(testDF[featurecols].values)
        else:
            testDF['predict_m'] =(np.array(morningmodel[0].predict_proba(testDF[featurecols].values)[:,1]) > morningmodel[1]).astype(int)
        if daymodel[1]==None:
            testDF['predict_d'] = daymodel[0].predict(testDF[featurecols].values)
        else:
            testDF['predict_d'] = (
            np.array(daymodel[0].predict_proba(testDF[featurecols].values)[:, 1]) > daymodel[1]).astype(int)
        testDF['predict'] = testDF.apply(
            lambda r: r['predict_m'] if r['firsttime'] < pd.to_timedelta('40m') else r['predict_d'], axis=1)
        datelist = testDF.date.unique()
        for i in range(len(paramset)):
            param=paramset[i]
            for date in datelist:
                thistrade = getPosition(testDF[testDF['date'] == date].copy(), param['revertwindow'], param['crashwindow'], param['crashthreshold'],
                                  param['reverthreshold'],param['usepredict'],param['stoploss'],param['stopgain'],param['timewindow'])
                alltrades[i] = alltrades[i] + [x + [stock, date] for x in thistrade]
                if (i in [19,22]) and len(thistrade)>0:
                    print(stock,date)
                    print([(np.round(x[0],4),x[1],str(x[2].time())) for x in thistrade])
    return alltrades

def calcFeatures(equitylist,param,featurecolsbase,featurecols,morningmodel,daymodel,startdate=pd.datetime(2017,2,3),enddate=pd.datetime(2017,11,29)):
    allFeatures=pd.DataFrame()
    # for stock in ['AAP']:
    for stock in equitylist:
        testDF = dataprocess.loadFeatureDataBulk(stock, featurecolsbase, temperAtOpen=[], truncPostMove=False,
                                                 startdate=startdate,enddate=enddate)
        if len(testDF)<1:
            print(stock,'load data fail')
            continue
        testDF['predict_m'] =np.array(morningmodel.predict_proba(testDF[featurecols].values)[:,1])
        testDF['predict_d'] = np.array(daymodel.predict_proba(testDF[featurecols].values)[:, 1])
        testDF['predict'] = testDF.apply(
            lambda r: r['predict_m'] if r['firsttime'] < pd.to_timedelta('40m') else r['predict_d'], axis=1)
        datelist = testDF.date.unique()

        for date in datelist:
            features = extractFeatures(testDF[testDF['date'] == date][['mid','predict','date','firsttime']+featurecols].copy(), param['crashwindow'],param['mincrash'],
                                       param['revertwindows'], param['timewindows'])
            features['stock']=stock

            allFeatures=pd.concat([allFeatures,features])
    return allFeatures


def extractFeatures(df,crashwindow,mincrash,revertwindows,timewindows,cutoffEOD=10):
    initial=df['mid'].values[0]
    df['diff']=df['mid'].diff()/initial
    df['max']=df['mid'].rolling(window=crashwindow).max()
    df['min']=df['mid'].rolling(window=crashwindow).min()
    df['range']=(df['max']-df['min'])/initial

    df['max_index']=df['mid'].rolling(crashwindow).apply(np.argmax)
    df['min_index'] = df['mid'].rolling(crashwindow).apply(np.argmin)
    df['crashsign']=np.sign(df['max_index']-df['min_index']) #positive for crash up,  negative for crash down
    df['revert']=df['crashsign']*(df['mid']-(df['crashsign']+1)/2*df['max']+(df['crashsign']-1)/2*df['min'])
    for revertwindow in revertwindows:
        df['signed_chg_'+str(revertwindow)]=df['mid'].diff(revertwindow)/initial*df['crashsign']
    df['max_predict']=df['predict'].rolling(window=crashwindow).max()
    df['avg_predict'] = df['predict'].rolling(window=crashwindow).mean()
    for revertwindow in timewindows:
        df['fut_signed_chg_'+str(revertwindow)]=(df['mid'].diff(revertwindow).shift(-revertwindow))/initial*df['crashsign']
    #flag has the sign of trade

    cutofftime=df['firsttime'][-1]-pd.to_timedelta(str(cutoffEOD)+'m')


    return df[(df['firsttime']<cutofftime) & (df['range']>mincrash) & (df['signed_chg_1']<=0)].\
        drop(['max_index','min_index'],axis=1).dropna()
