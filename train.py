import analytics.dataprocess as dataprocess
import analytics.modeltrain as modeltrain
import static
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df=pd.DataFrame()
equitylist=[x for x in static.equities_done if (x in static.equitties_train)]
for equity in equitylist:
    thisDF=dataprocess.loadFeatureData(equity,local=False)
    thisDF['stock']=equity
    df=pd.concat([df,thisDF])
df.dropna(inplace=True)

df=pd.DataFrame()
equitylist=[x for x in static.equities_done if (x in static.equitties_train)]
for equity in equitylist:
    thisDF=dataprocess.loadFeatureDataBulk(equity)
    thisDF['stock']=equity
    df=pd.concat([df,thisDF])
df.dropna(inplace=True)

stock='AAPL'
feature='litvolume'
startdate=pd.datetime(2017,2,10)
enddate=pd.datetime(2017,2,20)
tmp=pd.DataFrame()
for date in [x for x in df.date.unique() if ((pd.to_datetime(x)<enddate) and (pd.to_datetime(x)>=startdate))]:
    tmp[date]=df[(df['stock']==stock) & (df['date']==date)][feature].values
tmp.plot()


######crash method 1
crash_thrshld=0.02
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxchg_abs']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)

####crash method 2
crash_thrshld=0.025
#crash_thrshld=0.015
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxrange']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)
######summary
df['timebucket']=df.index.hour
print(df.groupby('timebucket')['target'].sum())


targetsummary=df.groupby(['stock','date'])['target'].max()
print(targetsummary.value_counts())
for equity in df['stock'].unique():
    x=targetsummary[equity].value_counts()
    y=df[df['stock'] == equity]['target'].value_counts()
    if 1 in x.index:
        print(equity+': by day %.4f, by minute %.4f'%(x[1]/x.sum(),(y[1] if 1 in y.index else 0)/y.sum()))
        #print(targetsummary[equity][targetsummary[equity] == 1].index)
    else:
        print(equity+': by day 0, by minute %.4f'% ((y[1] if 1 in y.index else 0)/y.sum()))


######train model
# load data


featurecols=['alltradeimb_auto','trfimb_auto','trfcntimb_auto','totimb_auto','mindepth','spread','litvolume']

# split data into train and test sets

print(pd.DataFrame({'largemv':df[df['maxchg_abs']>crash_thrshld].groupby('date')['maxchg_abs'].count(),\
                    'crash':df[df['target']==1].groupby('date')['target'].count()}).fillna(0))




kfold=5
seed=1

classifiers = [
    ('XGB_cus5', XGBClassifier(max_depth=5)),
    ('XGB_cus', XGBClassifier(max_depth=3)),
    ('XGB_cusR', XGBClassifier(max_depth=3,learning_rate=0.5)),
    ('XGB_cusS', XGBClassifier(max_depth=3,subsample=0.5)),
    ('XGB_cusD', XGBClassifier(max_depth=3,max_delta_step=3)),
    ('XGB_cusG', XGBClassifier(max_depth=3,gamma=10)),#bad
    ('XGB_cusA', XGBClassifier(max_depth=3,reg_alpha=10)),#bad
    ('XGB_cusL', XGBClassifier(max_depth=3, reg_lambda=100)),#bad
    ('DT5', DecisionTreeClassifier(max_depth=5)),
    ('DT3', DecisionTreeClassifier(max_depth=3))
]
kfoldds=0.01
dsratio=0.25
summaryDF=modeltrain.modelselect(df,featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=False)
print(summaryDF)

kfoldds=0.01
dsratio=0.1
minNeighbor=np.round(len(df)*(kfold-1)/kfold*dsratio*kfoldds*0.005).astype(int)
print("min neighbor: ",minNeighbor)
classifiers = \
    [('KN_1',KNeighborsClassifier(minNeighbor)),
    ('KN_1D',KNeighborsClassifier(minNeighbor,weights = 'distance')),
    ('KN_2',KNeighborsClassifier(minNeighbor*2)),
    ('KN_2D',KNeighborsClassifier(minNeighbor*2,weights = 'distance')),
    ('KN_4',KNeighborsClassifier(minNeighbor*4)),
    ('KN_4D',KNeighborsClassifier(minNeighbor*4,weights = 'distance'))]
#10D is the winner in this setup
summaryDF=modeltrain.modelselect(df,featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=True)
print(summaryDF)

summaryDF=modeltrain.modelselect(df,featurecols,classifiers,kfoldds,dsratio,kfold,seed,weightadj=False)
print(summaryDF)

#######diagonosis
targetstock='AFL'
model=XGBClassifier(max_depth=3,max_delta_step=3)
kfoldds=0.01
dsratio=0.25
newdf = modeltrain.modelDiagonose(df,targetstock,featurecols,model,kfoldds,dsratio,seed,weightadj=True)

#######diagonosis

#those don't work
classifiers = [
    ('RF',RandomForestClassifier(max_depth=5, n_estimators=10)),
    ('GNB',GaussianNB()),
    ('QDA',QuadraticDiscriminantAnalysis()),
    ('SVC_L',SVC(kernel="linear", C=0.025)),
    ('SVC_G',SVC(gamma=2, C=1)),
    ('LR_1',LogisticRegression(penalty='l1')),
    ('LR_2',LogisticRegression(penalty='l2'))
]
dsratio=0.05
summaryDF=modeltrain.modelselect(df,classifiers,0.1,dsratio,kfold,seed,weightadj=False)
print(summaryDF)



from sklearn.feature_selection import RFECV

allindex=np.array(range(len(X)))
major=allindex[Y==0]
minor=allindex[Y==1]
major_rs=resample(major, replace=False, n_samples=int(len(major) * 0.01),random_state=10)
X_rs=X[np.concatenate([major_rs, minor])]
Y_rs=Y[np.concatenate([major_rs, minor])]
# Create the RFE object and compute a cross-validated score.
model =  XGBClassifier(scale_pos_weight=1,max_delta_step=1,max_depth=3)

rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5,shuffle=True),
              scoring='accuracy')
rfecv.fit(X_rs, Y_rs)

print("Optimal number of features : %d" % rfecv.n_features_)



xgb.plot_tree(model)
