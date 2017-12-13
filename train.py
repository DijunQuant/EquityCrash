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


df=pd.DataFrame()

equitylist=[x for x in static.equities_done if (x in static.equitties_train)]
for equity in equitylist:
    thisDF=dataprocess.loadFeatureData(equity,local=False)
    thisDF['stock']=equity
    df=pd.concat([df,thisDF])
df.dropna(inplace=True)

######crash method 1
crash_thrshld=0.02
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxchg_abs']>crash_thrshld and r['eodchange']*r['maxchg']<reverseratio*r['maxchg']*r['maxchg']) else 0, axis=1)

####crash method 2
crash_thrshld=0.02
#crash_thrshld=0.015
reverseratio=0.5
df['target']=df.apply(lambda r: 1 if (r['maxrange']>crash_thrshld and np.abs(r['eodchange'])<reverseratio*r['maxrange']) else 0, axis=1)

######summary
targetsummary=df.groupby(['stock','date'])['target'].max()
print(targetsummary.value_counts())
for equity in equitylist:
    x=targetsummary[equity].value_counts()
    y=df[df['stock'] == equity]['target'].value_counts()
    if 1 in x.index:
        print(equity+': by day %.4f, by minute %.4f'%(x[1]/x.sum(),(y[1] if 1 in y.index else 0)/y.sum()))
        print(targetsummary[equity][targetsummary[equity] == 1].index)
    else:
        print(equity+': by day 0, by minute %.4f'% ((y[1] if 1 in y.index else 0)/y.sum()))



######train model
# load data


featurecols=['alltradeimb_auto','trfimb_auto','trfcntimb_auto','totimb_auto','mindepth']

# split data into train and test sets

print(pd.DataFrame({'largemv':df[df['maxchg_abs']>crash_thrshld].groupby('date')['maxchg_abs'].count(),\
                    'crash':df[df['target']==1].groupby('date')['target'].count()}).fillna(0))
def modelselect(df,classifiers,predsratio,dsratio,kfold,seed,weightadj=True):
    ####classifiers need resample
    result_dict = dict()
    kfsplits = modeltrain.stratifiedDF(kfold, seed, df,dsample=predsratio)
    for (name, model) in classifiers:
        if name in result_dict.keys(): continue
        accuracy_result, precision_result, cm, testDF = modeltrain.train_model(model, df, featurecols, kfsplits, dsratio,weightadj=weightadj)
        byday = testDF.groupby('target')['predict'].value_counts()
        score_0 = (byday[(0, 0)] if (0, 0) in byday.index else 0) / (byday[0].sum())
        score_1 = (byday[(1, 1)] if (1, 1) in byday.index else 0) / (byday[1].sum())
        result_dict[name] = (np.mean(precision_result), np.mean(accuracy_result), score_0, score_1)
        print(name+':%.3f, %.3f, %.3f, %.3f'% (np.mean(precision_result),np.mean(accuracy_result),score_0,score_1))
    summaryDF = pd.DataFrame.from_dict(result_dict, orient='index')
    summaryDF.columns = ['precision', 'accuracy', 'score_0', 'score_1']
    summaryDF.sort_values(['score_1', 'score_0'], ascending=False, inplace=True)
    return summaryDF

classifiers = [
    ('XGB',XGBClassifier(scale_pos_weight=10)),
    ('XGB_cus',XGBClassifier(scale_pos_weight=10,max_delta_step=1,max_depth=3)),
    ('XGB_cus1',XGBClassifier(scale_pos_weight=10,max_delta_step=1,max_depth=3,reg_alpha=1,gamma=1)),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    ('DT',DecisionTreeClassifier(max_depth=5,class_weight={0:0,1:10})),
    #('RF',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,class_weight='balanced'))
    #MLPClassifier(alpha=1),
    #AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
]

kfold=5
seed=7
dsratio=1
summaryDF=modelselect(df,classifiers,0.1,dsratio,kfold,seed)
print(summaryDF)



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
dsratio=0.05
summaryDF=modelselect(df,classifiers,0.1,dsratio,kfold,seed,weightadj=False)
print(summaryDF)


classifiers = \
    [('KN_5',KNeighborsClassifier(5)),
    ('KN_5D',KNeighborsClassifier(5,weights = 'distance')),
    ('KN_10',KNeighborsClassifier(10)),
    ('KN_10D',KNeighborsClassifier(10,weights = 'distance')),
    ('KN_20',KNeighborsClassifier(20)),
    ('KN_20D',KNeighborsClassifier(20,weights = 'distance'))]
#10D is the winner in this setup
dsratio=0.05
summaryDF=modelselect(df,classifiers,0.1,dsratio,kfold,seed,weightadj=True)
print(summaryDF)


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
summaryDF=modelselect(df,classifiers,0.1,dsratio,kfold,seed,weightadj=False)
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
