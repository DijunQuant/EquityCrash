import analytics.dataprocess as dataprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

equity = 'DLPH'

#####generate features
dataprocess.computeFeatureForEquity(equity,dataprocess.parameter,pd.datetime(2017,2,3),pd.datetime(2017,11,29),local=False,output_local=True)
#####generate features

####temp
equity='ZTS'
for file in os.listdir('Data//features//' + equity + '//'):
    thisdf = pd.DataFrame.from_csv('Data//features//' + equity + '//' + file)
    if len(thisdf) < 1: continue
    if os.path.exists('Data//bookfeature//'+equity+'//'+file):
        bookfeature = pd.DataFrame.from_csv('Data//bookfeature//'+equity+'//'+file)
        if len(bookfeature)==0:continue
    else: continue
    thisdf=thisdf.merge(bookfeature.drop(['date','mid'],axis=1),left_index=True,right_index=True)
    thisdf.to_csv('Data//features_tmp//' + equity + '//' + file)
####temp


df=dataprocess.loadFeatureData('KORS')
df.dropna(inplace=True)


print(len(df),len(df[np.abs(df['eodchange'])<df['maxchg_abs']*0.5]))
print(np.round(df[['maxchg','maxchg_abs','mindepth']+[x for x in df.columns if 'imb' in x]+[x for x in df.columns if 'all_' in x]].corr()
               [['maxchg','maxchg_abs']],3))
print(np.round(df[np.abs(df['eodchange'])<df['maxchg_abs']*0.5]\
                   [['maxchg','maxchg_abs','mindepth']+[x for x in df.columns if 'imb' in x]+[x for x in df.columns if 'all_' in x]].corr()
               [['maxchg','maxchg_abs']],3))

df[df['date']=='2017-06-12'][['all_bid','all_ask','alltradeimb_auto','totimb_auto','mid']].plot(secondary_y='mid')


df['maxchg_absQ']=pd.qcut(df['maxchg_abs'],10,labels=range(10))
df['eodchg_ratioQ']=df.apply(lambda r: 1 if r['eodchange']*r['maxchg']<0.5*r['maxchg']*r['maxchg'] else 0, axis=1)

df.boxplot(column=['alltradeimb_auto','trfimb_auto','trfcntimb_auto','totimb_auto'], by=['maxchg_absQ','eodchg_ratioQ'],layout=(4,1))


import seaborn as sns


g = sns.PairGrid(df, vars=["maxchg", "alltradeimb",'totimb','trfimb','trfcntimb'])
g = g.map(plt.scatter)
