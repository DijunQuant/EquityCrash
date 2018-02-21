import analytics.dataprocess as dataprocess
import analytics.modeltrain as modeltrain
import static
import os
import pandas as pd


import multiprocessing as mp


def loaddata(i,equities,featurecolsbase,resample_freq,return_dict):
    temp = pd.DataFrame()
    for equity in equities:
        print('run',equity)
        # thisDF=dataprocess.loadFeatureDataBulk(equity,featurecolsbase,temperAtOpen=[],resample=resample_freq,startdate=pd.datetime(2017,5,29))
        thisDF = dataprocess.loadFeatureDataBulk(equity, featurecolsbase, temperAtOpen=[], resample=resample_freq)
        if len(thisDF) > 0:
            thisDF['stock'] = equity
            temp = pd.concat([temp, thisDF])
    return_dict[i]=temp.dropna()
manager = mp.Manager()


#equitylist=[x for x in static.equities_done if (x in static.equitties_train)]
def main(largeCap,featurecolsbase):
    df=pd.DataFrame()
    return_dict = manager.dict()
    resample_freq = 1  # minute


    equitydict = dict(static.equities_train)

    if largeCap:
        equitylist = [x[0] for x in static.equities_train if
                      ((x[0] in static.equities_done) and (equitydict[x[0]] > static.median_cap_threshold))]
    else:
        equitylist = [x[0] for x in static.equities_train if
                      ((x[0] in static.equities_done) and (equitydict[x[0]] <= static.median_cap_threshold))]
    #equitylist = equitylist[:5]
    print(len(static.equities_done), len(equitylist))

    thread=8
    jobs = []
    for i in range(thread):
        p = mp.Process(target=loaddata, args=(i,equitylist[i::thread], featurecolsbase,resample_freq,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    for i in range(thread):
        df=pd.concat([df,return_dict[i]])
    return df

