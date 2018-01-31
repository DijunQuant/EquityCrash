import analytics.dataprocess as dataprocess
import os
import pandas as pd
import numpy as np
import sys
if sys.version_info < (3, 0):
    import matplotlib
    matplotlib.use('agg',warn=False, force=True)
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
def train_model(model, df,featurecols, kfsplits, rsratio_maj,seed=7,weightadj=False,return_prob=False,target_prob=None):
    # split data into X and y
    df_indexed=df.copy()
    df_indexed['ts']=df_indexed.index
    df_indexed=df_indexed.set_index(['stock','date']).sort_index(level=0)
    df_indexed['predict']=np.nan
    accuracy_result = []
    precision_result = []
    threshold_result=[]
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
        predictions = model.predict(X_test)
        predictions_prob=model.predict_proba(X_test)[:,1]
        threshold=0

        #predictions = [round(value) for value in y_pred]
        if return_prob:
            df_indexed.loc[test_index,'predict']= predictions_prob
        else:
            if target_prob != None:
                threshold = np.percentile(model.predict_proba(X_train)[:,1], target_prob)
                predictions = (np.array(predictions_prob)>threshold).astype(int)
                #print(threshold)
            df_indexed.loc[test_index, 'predict'] = predictions
        accuracy = accuracy_score(Y_test, predictions)
        accuracy_result.append(accuracy)
        threshold_result.append(threshold)
        cm = confusion_matrix(Y_test, predictions)
        cm_result.append(cm)
        precision_result.append(cm[1][1] / (cm[1][0] + cm[1][1]))
        # print(confusion_matrix(Y[test], predictions))
    return accuracy_result, precision_result, threshold_result,cm_result,df_indexed.dropna()

def modelselect(df,features,classifiers,predsratio,dsratio,kfold,seed,weightadj=True,target_prob=None):
    ####classifiers need resample
    result_dict = dict()
    kfsplits = stratifiedDF(kfold, seed, df,dsample=predsratio)
    alloosdf = df.reset_index()[['target', 'maxup', 'maxdown', 'maxrange', 'stock', 'date']]
    alloosdf['ts'] = df.index
    for (name, model) in classifiers:
        if name in result_dict.keys(): continue
        oosdf = pd.DataFrame()
        accuracy_result, precision_result, threshold_result,cm, testDF = train_model(model, df, features, kfsplits, dsratio,\
                                                                    weightadj=weightadj,target_prob=target_prob)
        oosdf = pd.concat([oosdf, testDF])
        oospredict=oosdf.groupby(['stock','date'])[['target','predict']].max()
        byday = oospredict.groupby('target')['predict'].value_counts()
        alarms = len(oospredict[oospredict['predict'] == 1])
        score_0 = (float(byday[(1, 1)] if (1, 1) in byday.index else 0) / (alarms)) if alarms>0 else 0 # chance of correct when predict positive
        score_1 = float(byday[(1, 1)] if (1, 1) in byday.index else 0) / (byday[1].sum()) #chance of not miss positive example
        result_dict[name] = (np.mean(precision_result), np.mean(accuracy_result), score_0, score_1,float(alarms)/len(oospredict))
        alloosdf = alloosdf.merge(
            pd.DataFrame({name: oosdf['predict'], 'ts': oosdf['ts'], 'stock': oosdf.index.get_level_values('stock')
                          }), on=['stock', 'ts'], how='left')
        print(name + ':%.3f, %.3f, %.3f, %.3f, %.3f' % (
        np.mean(precision_result), np.mean(accuracy_result), np.mean(threshold_result), score_0, score_1))
    summaryDF = pd.DataFrame.from_dict(result_dict, orient='index')
    summaryDF.columns = ['precision', 'accuracy', 'score_0', 'score_1','alarm']
    summaryDF.sort_values(['score_1', 'score_0'], ascending=False, inplace=True)
    return summaryDF,alloosdf

def modelselectbytime(df,timecut,features,classifiers,predsratio,dsratio,kfold,seed,weightadj=True,target_prob=None):
    result_dict = dict()
    thisdf = [df[df['firsttime'] < pd.to_timedelta(timecut)],df[df['firsttime'] >= pd.to_timedelta(timecut)]]
    kfsplits = [stratifiedDF(kfold, seed, thisdf[0], dsample=predsratio),stratifiedDF(kfold, seed, thisdf[1], dsample=predsratio)]
    alloosdf=df.reset_index()[['target','maxup','maxdown','maxrange','stock','date']]
    alloosdf['ts']=df.index

    for (name, model) in classifiers:
        if name in result_dict.keys(): continue
        oosdf=pd.DataFrame()
        try:
            for i in [0,1]:
                if type(dsratio)==float:
                    ds=dsratio
                else:
                    ds=dsratio[i]
                accuracy_result, precision_result, threshold_result,cm, testDF = train_model(model, thisdf[i], features, kfsplits[i], ds,\
                                                                    weightadj=weightadj,target_prob=target_prob)
                oosdf=pd.concat([oosdf,testDF])
        except:
            continue
        alloosdf=alloosdf.merge(pd.DataFrame({name:oosdf['predict'],'ts':oosdf['ts'],'stock':oosdf.index.get_level_values('stock')
}),on=['stock','ts'],how='left')
        oospredict=oosdf.groupby(['stock','date'])[['target','predict']].max()
        byday = oospredict.groupby('target')['predict'].value_counts()
        alarms = len(oospredict[oospredict['predict'] == 1])
        score_0 = float(((byday[(1, 1)] if (1, 1) in byday.index else 0)) / float(alarms)) if alarms>0 else 0 # chance of correct when predict positive
        score_1 = float(byday[(1, 1)] if (1, 1) in byday.index else 0) / float(byday[1].sum()) #chance of not miss positive example
        result_dict[name] = (np.mean(precision_result), np.mean(accuracy_result), score_0, score_1,float(alarms)/float(len(oospredict)))
        print(name+':%.3f, %.3f, %.3f, %.3f, %.3f'% (np.mean(precision_result),np.mean(accuracy_result),np.mean(threshold_result),score_0,score_1))
    summaryDF = pd.DataFrame.from_dict(result_dict, orient='index')
    summaryDF.columns = ['precision', 'accuracy', 'score_0', 'score_1','alarm']
    summaryDF.sort_values(['score_1', 'score_0'], ascending=False, inplace=True)
    return summaryDF,alloosdf

def modelfitbytime(df,timecut,features,model,predsratio,dsratio,kfold,seed,weightadj=True,target_prob=None):
    #timecut = '30m'
    ####classifiers need resample
    result_dict = dict()
    thisdf = [df[df['firsttime'] < pd.to_timedelta(timecut)],df[df['firsttime'] >= pd.to_timedelta(timecut)]]
    kfsplits = [stratifiedDF(kfold, seed, thisdf[0], dsample=predsratio),stratifiedDF(kfold, seed, thisdf[1], dsample=predsratio)]
    oosdf=pd.DataFrame()
    for i in [0,1]:
        if type(dsratio) ==float:
            ds = dsratio
        else:
            ds = dsratio[i]
        accuracy_result, precision_result, threshold_result,cm, testDF = train_model(model, thisdf[i], features, kfsplits[i], ds,\
                                                                    weightadj=weightadj,target_prob=target_prob)
        oosdf=pd.concat([oosdf,testDF])
    return oosdf

def modelfit(df,features,model,predsratio,dsratio,seed,weightadj=False,target_prob=None):
    targetsummary = df.groupby(['stock', 'date'])['target'].max()
    majority = targetsummary.index[targetsummary == 0]
    minority = targetsummary.index[targetsummary == 1]
    if predsratio < 1:
        majority = resample(majority, replace=False, n_samples=int(len(majority) * predsratio),
                            random_state=seed)
    train=np.concatenate([majority,minority])

    train_data = df.set_index(['stock','date']).sort_index(level=0).loc[train][['target','maxchg_abs'] + features]
    if weightadj:
        wts = train_data.corr()['maxchg_abs']
        train_data[features] = train_data[features].mul(wts[features], axis=1)

    train_minority = train_data[train_data['target'] == 1]
    train_majority = train_data[train_data['target'] == 0]
    if dsratio < 1:
        train_majority = resample(train_majority, replace=False, n_samples=int(len(train_majority) * dsratio),
                                  random_state=seed)
    # train_minus = resample(train_minority, replace=True, n_samples=int(len(train_minority) * rsratio_min),
    #                       random_state=seed)
    X_train = np.concatenate([train_majority[features].values, train_minority[features].values])
    Y_train = np.concatenate([train_majority['target'].values, train_minority['target'].values])
    model.fit(X_train, Y_train)
    if target_prob==None:
        return None
    else:
        return np.percentile(model.predict_proba(X_train)[:,1], target_prob)


def modelDiagonose(df,targetstock,features,model,predsratio,dsratio,seed,weightadj=True,target_prob=None,timecutoff=None):
    ####classifiers need resample
    if timecutoff==None:
        testDF=modelDiagonoseWork(df,targetstock,features,model,predsratio,dsratio,seed,weightadj=weightadj,target_prob=target_prob)
    else:
        testDF = pd.concat([
            modelDiagonoseWork(df[df['firsttime'] < pd.to_timedelta(timecutoff)], targetstock, features, model, predsratio, dsratio, seed, weightadj=weightadj,
                                    target_prob=target_prob),
            modelDiagonoseWork(df[df['firsttime'] >= pd.to_timedelta(timecutoff)], targetstock, features, model,
                               predsratio, dsratio, seed, weightadj=weightadj,
                               target_prob=target_prob)
        ])
    featureDF = dataprocess.loadFeatureDataBulk(targetstock,truncPostMove=False)

    commonfeature=[feature for feature in features if feature in featureDF.columns]

    return featureDF[commonfeature+['mid']].merge(testDF.set_index('ts')[['predict','target','date']],left_index=True,right_index=True,how='left')\
        .fillna(0)


def modelDiagonoseWork(df,targetstock,features,model,predsratio,dsratio,seed,weightadj=True,target_prob=None):
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
    accuracy_result, precision_result, threshold_result, cm, testDF = train_model(model, df, features, [(train,test)], dsratio,
                                                                weightadj=weightadj,target_prob=target_prob)
    print(accuracy_result,precision_result)
    print(cm)
    testDF.reset_index(level=1,inplace=True)

    summaryDF=testDF.groupby('date')[['predict','target']].max()
    print(summaryDF.reset_index().groupby(['predict','target'])['date'].count())
    return testDF

