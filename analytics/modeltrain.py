import analytics.dataprocess as dataprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.utils import resample

from sklearn.model_selection import StratifiedKFold

#df must have stock,date,target field
def stratifiedDF(n_split,seed,df,dsample=1):
    targetsummary = df.groupby(['stock', 'date'])['target'].max()
    kfsplits=[]
    for train,test in StratifiedKFold(n_splits=n_split, shuffle=True, \
                                      random_state=seed).split(np.zeros(len(targetsummary)), targetsummary.values):
        majority=train[targetsummary.iloc[train]==0]
        minority=train[targetsummary.iloc[train]==1]
        if dsample<1:
            majority = resample(majority, replace=False, n_samples=int(len(majority) * dsample),
                               random_state=seed)
        kfsplits.append((targetsummary.iloc[np.concatenate([majority,minority])].index,targetsummary.iloc[test].index))
        #kfsplits.append((df_indexed.loc[targetsummary.iloc[train].index].reset_index()))
    return kfsplits

#featurecols = ['alltradeimb_auto', 'trfimb_auto', 'trfcntimb_auto', 'totimb_auto', 'mindepth']
def train_model(model, df,featurecols, kfsplits, rsratio_maj,seed=7,weightadj=False):
    # split data into X and y
    df_indexed=df.copy()
    df_indexed['ts']=df_indexed.index
    df_indexed=df_indexed.set_index(['stock','date']).sort_index(level=0)
    df_indexed['predict']=np.nan
    accuracy_result = []
    precision_result = []
    cm_result = []
    for train_index, test_index in kfsplits:
        #print(wts[featurecols])
        train_data=df_indexed.loc[train_index][['target']+featurecols]
        if weightadj:
            wts = (df_indexed.loc[train_index][['maxchg_abs'] + featurecols].corr()['maxchg_abs'])
            train_data[featurecols]=train_data[featurecols].mul(wts[featurecols],axis=1)
            X_test = (df_indexed.loc[test_index][featurecols].mul(wts[featurecols],axis=1)).values
        else:
            X_test = (df_indexed.loc[test_index][featurecols]).values
        Y_test = df_indexed.loc[test_index]['target'].values
        train_minority = train_data[train_data['target'] == 1]
        train_majority = train_data[train_data['target'] == 0]
        if rsratio_maj<1:
            train_majority = resample(train_majority, replace=False, n_samples=int(len(train_majority) * rsratio_maj),
                               random_state=seed)
        #train_minus = resample(train_minority, replace=True, n_samples=int(len(train_minority) * rsratio_min),
        #                       random_state=seed)
        X_train=np.concatenate([train_majority[featurecols].values,train_minority[featurecols].values])
        Y_train = np.concatenate([train_majority['target'].values, train_minority['target'].values])

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        #predictions = [round(value) for value in y_pred]
        predictions = y_pred
        df_indexed.loc[test_index,'predict']=predictions
        accuracy = accuracy_score(Y_test, predictions)
        accuracy_result.append(accuracy)
        cm = confusion_matrix(Y_test, predictions)
        cm_result.append(cm)
        precision_result.append(cm[1][1] / (cm[1][0] + cm[1][1]))
        # print(confusion_matrix(Y[test], predictions))
    return accuracy_result, precision_result, cm_result,df_indexed.dropna()

def modelselect(df,features,classifiers,predsratio,dsratio,kfold,seed,weightadj=True):
    ####classifiers need resample
    result_dict = dict()
    kfsplits = stratifiedDF(kfold, seed, df,dsample=predsratio)
    for (name, model) in classifiers:
        if name in result_dict.keys(): continue
        accuracy_result, precision_result, cm, testDF = train_model(model, df, features, kfsplits, dsratio,weightadj=weightadj)
        oospredict=testDF.groupby(['stock','date'])[['target','predict']].max()
        byday = oospredict.groupby('target')['predict'].value_counts()
        score_0 = (byday[(0, 0)] if (0, 0) in byday.index else 0) / (byday[0].sum())
        score_1 = (byday[(1, 1)] if (1, 1) in byday.index else 0) / (byday[1].sum())
        result_dict[name] = (np.mean(precision_result), np.mean(accuracy_result), score_0, score_1)
        print(name+':%.3f, %.3f, %.3f, %.3f'% (np.mean(precision_result),np.mean(accuracy_result),score_0,score_1))
    summaryDF = pd.DataFrame.from_dict(result_dict, orient='index')
    summaryDF.columns = ['precision', 'accuracy', 'score_0', 'score_1']
    summaryDF.sort_values(['score_1', 'score_0'], ascending=False, inplace=True)
    return summaryDF

def modelDiagonose(df,targetstock,features,model,predsratio,dsratio,seed,weightadj=True):
    ####classifiers need resample
    traindata=df[df['stock']!=targetstock]
    test=df[df['stock']==targetstock].groupby(['stock', 'date'])['target'].max().index

    if len(test)<1:
        print('target not found:',targetstock)
        return
    targetsummary = traindata.groupby(['stock', 'date'])['target'].max()
    majority = targetsummary.index[targetsummary == 0]
    minority = targetsummary.index[targetsummary == 1]
    if predsratio < 1:
        majority = resample(majority, replace=False, n_samples=int(len(majority) * predsratio),
                            random_state=seed)
    train=np.concatenate([majority,minority])
    accuracy_result, precision_result, cm, testDF = train_model(model, df, features, [(train,test)], dsratio,
                                                                weightadj=weightadj)
    print(accuracy_result,precision_result)
    print(cm)
    testDF.reset_index(level=1,inplace=True)

    summaryDF=testDF.groupby('date')[['predict','target']].max()
    print(summaryDF.reset_index().groupby(['predict','target'])['date'].count())
    #print(summaryDF[summaryDF.max(axis=1)>0])
    featureDF = dataprocess.loadFeatureDataBulk(targetstock,truncPostMove=False)

    return featureDF[features+['mid']].merge(testDF.set_index('ts')[['predict','target','date']],left_index=True,right_index=True,how='left')\
        .fillna(0)


