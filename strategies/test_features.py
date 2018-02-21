import analytics.dataprocess as dataprocess
import analytics.modeltrain as modeltrain
import analytics.strategy as strategy
import static
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

pd.set_option('display.float_format', lambda x: '%.4f' % x)
import matplotlib.pyplot as plt

featurecolsbase=['alltradeimb_auto','trfimb_auto','trfcntimb_auto','totimb_auto','mindepthNYSE', 'mindepthNASDAQ', 'mindepthBATS','spread','litvolume']

resample_freq=1 #minute
df=pd.DataFrame()

equitydict=dict(static.equities_train)

equitylist=[x[0] for x in static.equities_train if ((x[0] in static.equities_done) and (equitydict[x[0]]>static.median_cap_threshold))]

#equitylist=equitylist[:6]

seed=1
equity_train=resample(equitylist, replace=False, n_samples=int(len(equitylist) * 0.66),
                            random_state=seed)
equity_test=[x for x in equitylist if (x not in equity_train)]

#equitylist=[x[0] for x in static.equities_train if ((x[0] in static.equities_done) and (equitydict[x[0]]<static.median_cap_threshold))]
print(len(static.equities_done),len(equitylist),len(equity_train),len(equity_test))

for equity in [x for x in equity_train]:
    thisDF=dataprocess.loadFeatureDataBulk(equity,featurecolsbase,temperAtOpen=[],resample=resample_freq)
    if len(thisDF)>0:
        thisDF['stock']=equity
        print(equity)
        df=pd.concat([df,thisDF])

df.dropna(inplace=True)
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

kfoldds=0.05 #for strategy
dsratio=0.2

dsratio_day=0.1
dsratio_morning=0.3
target_prob=None

#dsratio_day=dsratio
#dsratio_morning=dsratio
#target_prob=99


morningmodel=XGBClassifier(max_depth=3,subsample=0.5)
daymodel=XGBClassifier(max_depth=3,subsample=0.5)

#rewrite fit model with all data

day_threshold=modeltrain.modelfit(df[df['firsttime'] >= pd.to_timedelta('40m')], featurecols, daymodel, kfoldds, dsratio_day, seed, weightadj=False,target_prob=target_prob)
morning_threshold = modeltrain.modelfit(df[df['firsttime'] < pd.to_timedelta('40m')], featurecols, morningmodel, kfoldds, dsratio_morning, seed, weightadj=False,target_prob=target_prob)
print(morning_threshold,day_threshold)

df=[]

param={'crashwindow':30,'mincrash':0.015,'revertwindows':[1,3,5],'timewindows':[5,10,30]}
allfeatures = strategy.calcFeatures(equity_test,param,featurecolsbase,featurecols,morningmodel,daymodel)

featurelabel=['revert','range','max_predict','avg_predict']+['signed_chg_'+str(x) for x in param['revertwindows']]

allfeatures[featurelabel+['fut_signed_chg_'+str(x) for x in param['timewindows']]].describe()
allfeatures[featurelabel+['fut_signed_chg_'+str(x) for x in param['timewindows']]].corr()[['fut_signed_chg_'+str(x) for x in param['timewindows']]]

#largecap
allfeatures.to_csv('strategies/allfeatures.csv')


#remove clustering
crashthd=0.0165
min_interval=pd.to_timedelta('30m')
allfeaturesF=pd.DataFrame()
for name,data in allfeatures[allfeatures['range']>crashthd].groupby(['stock','date']):
    #print(data[featurelabel+['fut_signed_chg_'+str(x) for x in param['timewindows']]])
    ts=data.index
    temp=pd.DataFrame()
    while len(ts)>0:
        currenttime=ts[0]
        temp=pd.concat([temp,data.loc[ts[:1]]])
        ts=ts[ts-currenttime>min_interval]
    allfeaturesF=pd.concat([allfeaturesF,temp])
allfeaturesF.reset_index(inplace=True)


trademodel=DecisionTreeClassifier(max_depth=5)
trademodel=DecisionTreeClassifier(max_depth=3)
trademodel =XGBClassifier(max_depth=3,max_delta_step=3,learning_rate=0.5)
trademodel=LogisticRegression(penalty='l1')
trademodel =SVC(gamma=2, C=1,class_weight='balanced')


featurelabel=['range','max_predict','avg_predict']+['signed_chg_'+str(x) for x in param['revertwindows']]

featurelabel=['max_predict','avg_predict']

targetlabel='fut_signed_chg_10'
allfeaturesF['target']=(np.sign(-allfeaturesF[targetlabel]-0.001)+1)/2
allfeaturesF['predict']=np.nan
n_split=10
for train,test in StratifiedKFold(n_splits=n_split, shuffle=True, \
                                      random_state=seed).split(np.zeros(len(allfeaturesF)), allfeaturesF['target'].values):
    print(len(train),len(test))
    X_train=allfeaturesF.iloc[train][featurelabel].values
    Y_train=allfeaturesF.iloc[train]['target'].values
    trademodel.fit(X_train, Y_train)
    allfeaturesF.loc[test,'predict']=trademodel.predict((allfeaturesF.iloc[test][featurelabel]).values)

