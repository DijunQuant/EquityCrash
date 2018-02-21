import analytics.dataprocess as dataprocess
import analytics.modeltrain as modeltrain
import static
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import train_helper
import gc

featurecolsbase = ['alltradeimb_auto', 'trfimb_auto', 'trfcntimb_auto', 'totimb_auto', 'mindepthNYSE',
                   'mindepthNASDAQ', 'mindepthBATS', 'spread', 'litvolume']
largeCap=False


df = train_helper.main(largeCap,featurecolsbase)
gc.collect()
#df.to_csv('Data/features/tmp_20180109.csv')


#df=pd.read_csv('Data/features/tmp_20180105.csv',index_col=0)
#df.index=pd.to_datetime(df.index)

#check diurnal effect
#stock='AAL'
#feature='spread'
#startdate=pd.datetime(2017,2,10)
#enddate=pd.datetime(2017,2,20)
#tmp=pd.DataFrame()
#for date in [x for x in df.date.unique() if ((pd.to_datetime(x)<enddate) and (pd.to_datetime(x)>=startdate))]:
#    tmp[date]=df[(df['stock']==stock) & (df['date']==date)][feature].values
#tmp.plot()
#check diurnal effect

######crash method 1
df = df[df['firsttime']>pd.to_timedelta('3m')]

crash_thrshld=(0.0175 if largeCap else 0.02)
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxchg_abs']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)

####crash method 2
#crash_thrshld=0.025
#crash_thrshld=0.015
#reverseratio=0.5
#df['target']=df.apply(lambda r: 1 if (r['maxrange']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)
######summary
df['timebucket']=df.index.hour+np.round(df.index.minute/15)*0.25
print(df.groupby('timebucket')['target'].sum())


targetsummary=df.groupby(['stock','date'])['target'].max()
print(targetsummary.value_counts())
#for equity in df['stock'].unique():
#    x=targetsummary[equity].value_counts()
#    if 1 in x.index:
#        print(equity,targetsummary[equity][targetsummary[equity] == 1].index)


featurecols = featurecolsbase+[feature+'_1' for feature in featurecolsbase] +[feature+'_2' for feature in featurecolsbase]

kfold=5
seed=1

classifiers = [
    ('XGB_cus5', XGBClassifier(max_depth=5)),
    ('XGB_cus', XGBClassifier(max_depth=3)),
    ('XGB_cusR', XGBClassifier(max_depth=3,learning_rate=0.5)),
    ('XGB_cusS', XGBClassifier(max_depth=3,subsample=0.5)),#overall winner
    #('XGB_cusD', XGBClassifier(max_depth=3,max_delta_step=3)),
    ('DT5', DecisionTreeClassifier(max_depth=5)),
    ('RF', RandomForestClassifier()),
]
#classifiers = [
#    ('XGB_cusS', XGBClassifier(max_depth=3,subsample=0.5)),
#]


def modelselectwraper(classifiers,kfoldds,dsratio,target_prob):
    modelnames = [x[0] for x in classifiers]
    print(modelnames)
    #summaryDF=modeltrain.modelselect(df,featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=False,target_prob=49.9)
    summaryDF,oosDF=modeltrain.modelselectbytime(df,'40m',featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=False,target_prob=target_prob)
    #print(summaryDF)
    temp=[]
    temp.append(oosDF[['maxrange']].describe()['maxrange'])
    for x in modelnames:
        temp.append(oosDF[oosDF[x]>0][['maxrange']].describe()['maxrange'])
    oosDF['mega']=(oosDF[modelnames].mean(axis=1)>0.5).astype(int)
    temp.append(oosDF[oosDF['mega']>0][['maxrange']].describe()['maxrange'])
    #print(pd.DataFrame(temp,index=['all']+modelnames+['mega']))
    d=dict()
    oosDFbyday=oosDF.groupby(['stock','date'])[modelnames+['mega','target']].max()
    n=len(oosDFbyday)
    for name in modelnames+['mega']:
        t=oosDFbyday.groupby([name,'target'])['target'].count()
        alarm=t[1].sum()
        score_0=float(t[1,1] if (1,1) in t.index else 0)/alarm
        score_1=float(t[1,1] if (1,1) in t.index else 0)/((t[0,1] if (0,1) in t.index else 0)+float(t[1,1] if (1,1) in t.index else 0))
        d[name]=[score_0,score_1,float(alarm)/n]
    d = pd.DataFrame.from_dict(d,orient='index')
    d.columns=['score_0','score_1','alarm']
    print(d)
    return


kfoldds=0.05 #for strategy
dsratio=0.2

dsratio_day=0.1
dsratio_morning=0.3


print('morning:')
print(df[df['firsttime'] < pd.to_timedelta('40m')].groupby(['stock','date'])['target'].max().value_counts())
#for targetp in []:
for targetp in [None,98,99]:
    print(targetp)
    summaryDF_morning,oosDF=modeltrain.modelselect(df[df['firsttime'] < pd.to_timedelta('40m')],featurecols,classifiers,kfoldds,dsratio_morning,kfold,seed,weightadj=False,target_prob=targetp)
    print(summaryDF_morning)
print('day')
print(df[df['firsttime'] >= pd.to_timedelta('40m')].groupby(['stock','date'])['target'].max().value_counts())
#for targetp in  []:
for targetp in [None,98,99]:
    print(targetp)
    summaryDF_day,oosDF=modeltrain.modelselect(df[df['firsttime'] >= pd.to_timedelta('40m')],featurecols,classifiers,kfoldds,dsratio_day,kfold,seed,weightadj=False,target_prob=targetp)
    print(summaryDF_day)


for targetp in [None,98,98.5,99]:
#for targetp in []:
    print(targetp)
    #modelselectwraper(classifiers,kfoldds,dsratio,targetp)
    modelselectwraper(classifiers, kfoldds, (dsratio_morning,dsratio_day), targetp)


runKNN=False
if runKNN:
    kfoldds=0.05
    dsratio=0.5
    minNeighbor=np.round(len(df)*(kfold-1)/kfold*dsratio*kfoldds*0.00005).astype(int)
    print("min neighbor: ",minNeighbor)
    knnclassifiers = \
        [('KN_1',KNeighborsClassifier(minNeighbor)),
        ('KN_1D',KNeighborsClassifier(minNeighbor,weights = 'distance')),
        ('KN_2',KNeighborsClassifier(minNeighbor*2)),
        ('KN_2D',KNeighborsClassifier(minNeighbor*2,weights = 'distance')),
        ('KN_4',KNeighborsClassifier(minNeighbor*4)),
        ('KN_4D',KNeighborsClassifier(minNeighbor*4,weights = 'distance'))]
    #10D is the winner in this setup
    summaryDF,oosDF_KN=modeltrain.modelselectbytime(df,'40m',featurecolsbase,knnclassifiers,kfoldds,dsratio,kfold,seed,weightadj=True,target_prob=49.95)
    print(summaryDF)

    summaryDF=modeltrain.modelselect(df,featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=False)
    print(summaryDF)

#######diagonosis

#targetstock='ADSK'
#model=XGBClassifier(max_depth=3,subsample=0.5)
#kfoldds=0.05
#dsratio=0.25
#seed=5
#newdf = modeltrain.modelDiagonose(df,targetstock,featurecols,model,kfoldds,dsratio,seed,weightadj=False,target_prob=49.9)
#print(list(zip(featurecols,model.feature_importances_)))
#print(model.get_score(importance_type='gain'))
#summary=newdf.groupby('date')[['target','predict']].max()
#print(summary[summary['target']!=summary['predict']])

#newdf['bucket']=newdf.index.hour+np.floor(newdf.index.minute/15)*0.25
#print(newdf.groupby('bucket')[['target','predict']].sum())

#######diagonosis

#those don't wor
#classifiers = [
#    ('RF',RandomForestClassifier(max_depth=5, n_estimators=10)),
#    ('GNB',GaussianNB()),
#    ('QDA',QuadraticDiscriminantAnalysis()),
#    ('SVC_L',SVC(kernel="linear", C=0.025)),
#    ('SVC_G',SVC(gamma=2, C=1)),
#    ('LR_1',LogisticRegression(penalty='l1')),
#    ('LR_2',LogisticRegression(penalty='l2'))
#]
#dsratio=0.05
#summaryDF=modeltrain.modelselect(df,classifiers,0.1,dsratio,kfold,seed,weightadj=False)
#print(summaryDF)


#df has 'target' and 'predict' colume
def getROCcurve(df,step):
    poscase=df[df['target']==1]['predict'].copy()
    def errorprob(x):
        tmp=df[df['predict']>x]['target'].value_counts()
        return tmp[0]/tmp.sum() if 0 in tmp.keys() else 0
    def missprob(x):
        return len(poscase[poscase<x])/len(poscase)
    xlist=np.arange(0,1,step)
    return list(xlist),[len(df[df['predict']>x])/len(df) for x in xlist],[errorprob(x) for x in xlist], [missprob(x) for x in xlist]
