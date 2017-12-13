import numpy as np
import pandas as pd
import s3fs
import os
from scipy import interpolate
import multiprocessing as mp
import static


fs = s3fs.S3FileSystem()

venue=['NYSE_MKT','NASDAQ_BX','BATS_EDGA','BATS_EDGX','CHX','NYSE','NYSE_ARCA',\
       'NASDAQ','IEX','NASDAQ_PSX','BATS_BYX','BATS_BZX']
localroot='Data//'
s3rootraw='s3://eastend-flash/flash/'
s3root='s3://equity-flash/'

#featurefolder='Data//features//'


datacolume=['ts','bidpx','askpx','buy','sell','other','trf','buyamt','sellamt','otheramt','trfamt']\
            +[x+'_bid' for x in venue]+[x+'_ask' for x in venue]+['buytot','selltot']
spreadthreshold = 0.0025
exclFirstNRow=60
exclLastNRow=60

###
parameter={'crash_in_second':60*30, #to reach peak/bottom
           'rolling_window':60*5,#
           'autocorr_bin':10,
           'autocorr_window':30,
           'sample_interval':60,
           'bookrank_in_day':5
            }
####
def computeFeatureForEquities(num,local,parallel=True,output_local=None):
    to_run=[name for name in static.equitties_train if (name not in static.equities_done)]
    if len(to_run)<num: return

    if parallel:
        processes = [
            mp.Process(target=computeFeatureForEquity,\
                   args=(equity,parameter, pd.datetime(2017,2,3),pd.datetime(2017,11,29),local,output_local))\
            for equity in to_run[:num]]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
            print(p.pid,'done')
    else:
        for equity in to_run[:num]:
            computeFeatureForEquity(equity,parameter, pd.datetime(2017,2,3),pd.datetime(2017,11,29),local,output_local)

def computeFeatureForEquity(equity, param, startdate,enddate,local=True,output_local=None):
    #hdf = pd.HDFStore(folder+'features//'+equity+'.h5')
    #output=pd.DataFrame()
    #df = pd.DataFrame(columns=['date', 'range', 'change']).set_index('date')
    histbookdata =pd.DataFrame(columns=['date']+[site + 'all_bid' for site in['','NYSE', 'NASDAQ', 'BATS']]+\
                                       [site + 'all_ask' for site in['','NYSE', 'NASDAQ', 'BATS']])
    datelist=[]
    if output_local==None: output_local=local
    if output_local:
        outputpath = localroot
        if not os.path.exists(outputpath + 'features//' + equity + '//'):
            os.mkdir(outputpath + 'features//' + equity + '//')
    else:
        outputpath = s3root
        #if not fs.exists(outputpath + 'features//' + equity + '//'):
        #    fs.mkdir(outputpath + 'features//' + equity + '//')

    if local:
        prefixpath=localroot+equity

        for file in os.listdir(prefixpath):
            if file.startswith('.'): continue
            filenameparser=file.split('.')[0].split('-')
            thisdate=pd.datetime(int(filenameparser[1]),int(filenameparser[2]),int(filenameparser[3]))
            datelist.append(thisdate)
    else:
        prefixpath=s3rootraw+equity
        #outputpath=os.path.expanduser('~')+'//Data//'
        for file in fs.ls(prefixpath+'//'):
            if file.startswith('.'): continue
            filenameparser = file.split('/')[-1].split('.')[0].split('-')
            thisdate = pd.datetime(int(filenameparser[1]), int(filenameparser[2]), int(filenameparser[3]))
            datelist.append(thisdate)
    datelist=sorted(datelist)
    print(equity,len(datelist),datelist[0],datelist[-1])

    for thisdate in datelist:
        if thisdate<startdate or thisdate>enddate:
            print(thisdate,' skipped')
            continue
        file=equity+'-'+thisdate.strftime('%Y-%m-%d')+'.csv.gz'
        print(thisdate, len(histbookdata), prefixpath + '/' + file)

        df = getDataAndFilter(prefixpath + '/' + file).reset_index()
        if len(df) > 0:
            df['date'] = thisdate.date()
            for side in ['_ask', '_bid']:
                df['all' + side] = df[[x + side for x in venue]].sum(axis=1)
                for major in ['NYSE', 'NASDAQ', 'BATS']:
                    df[major + 'all' + side] = df[[x + side for x in venue if major in x]].sum(axis=1)
                df = df.drop([x + side for x in venue], axis=1)
            outputfile=outputpath + 'features/' + equity + '/' + thisdate.strftime('%Y%m%d') + '.csv'

            if output_local:
                if not os.path.exists(outputfile):
                    computeFeatures(df, thisdate, param, histbookdata).to_csv(outputpath + 'features//' + equity + '//' + thisdate.strftime('%Y%m%d') + '.csv')
            else:
                #print('check exist',outputfile,fs.exists(outputfile))
                if not fs.exists(outputfile):
                    bytes_to_write = computeFeatures(df, thisdate, param, histbookdata).to_csv(None).encode()

                    with fs.open(outputfile, 'wb') as f:
                        f.write(bytes_to_write)
            histbookdata = updatehistbookdata(df,param,histbookdata)


def updatehistbookdata(df,param,histbookdata):
    if len(histbookdata['date'].unique())>=param['bookrank_in_day']:
        mindate=sorted(histbookdata['date'].unique())[0]
        histbookdata= histbookdata[histbookdata.date>mindate]
        #print(thisdate, 'drop ' + str(mindate),len(histbookdata))
    #update histbookdata

    return pd.concat([histbookdata,df[['date']+[site + 'all_bid' for site in['','NYSE', 'NASDAQ', 'BATS']]+\
                                             [site + 'all_ask' for site in['','NYSE', 'NASDAQ', 'BATS']]]])

def computeFeatures(df,thisdate,param,histbookdata):
    bookfeatures = computeBookFeature(df,thisdate,param,histbookdata)
    if len(bookfeatures)<1: return pd.DataFrame()
    calcImb(df)
    df['alltradeimb']=df['tradeimb']+df['otherimb']
    df.index=pd.to_datetime(thisdate)+pd.to_timedelta(df['ts'])
    first=df['mid'].iloc[0]
    last=df['mid'].iloc[-1]
    #compute max move from now to end of day
    forwardwindow=param['crash_in_second']
    result=pd.DataFrame({'running_min':[df.iloc[i:i+forwardwindow]['mid'].min() for i in range(len(df)-forwardwindow)],\
                         'running_max':[df.iloc[i:i+forwardwindow]['mid'].max() for i in range(len(df)-forwardwindow)],\
                         'mid':df.iloc[:-forwardwindow]['mid'],\
                         'ts':df.iloc[:-forwardwindow]['ts'],\
                         'date':thisdate})
    result['maxup']=(result['running_max']-result['mid'])/first
    result['maxdown']=(result['running_min']-result['mid'])/first
    result['eodchange']=(last-result['mid'])/first
    result.index = pd.to_datetime(result['date']) + pd.to_timedelta(result['ts'])
    result.drop('ts',axis=1,inplace=True)
    result=result.merge(df[['alltradeimb','totimb','trfimb','trfcntimb']].rolling(param['rolling_window']).mean(),how='left',\
                        left_index=True,right_index=True)
    autocorr_df = df.groupby(pd.TimeGrouper(freq=str(param['autocorr_bin']) + 'S'))['trfimb','trfcntimb', 'alltradeimb', 'totimb'].sum().\
        rolling(param['autocorr_window']).apply(lambda x: pd.Series(x).autocorr(1))
    result=pd.merge_asof(result,autocorr_df, left_index=True,right_index=True, suffixes=('', '_auto'))
    result = result.merge(bookfeatures,how='left',left_index=True,right_index=True)

    result=result.dropna().resample(str(param['sample_interval'])+'S').last()
    return result
def computeBookFeature(df,thisdate,param,histbookdata):
    result=pd.DataFrame()

    def interprank(key):
        ylist = np.arange(0.05, 1, 0.05)
        xlist = histbookdata[key].quantile(ylist).values
        #print(df[key].values)
        result[key] = np.interp(np.array(df[key].values,dtype='float64'), \
                                np.array(xlist,dtype='float64'), np.array(ylist,dtype='float64'), left=0., right=1.)

        #f = interpolate.interp1d(xlist, ylist, fill_value=(0, 1))
        #result[key] = list(map(f, df[key].values))
        result[key]= result[key].rolling(param['rolling_window']).mean()

    if len(histbookdata['date'].unique())>=param['bookrank_in_day']:
        for side in ['_ask', '_bid']:
            for site in ['','NYSE', 'NASDAQ', 'BATS']:
                interprank(site + 'all' + side)

        result['date']= thisdate
        result.index = pd.to_datetime(result['date']) + pd.to_timedelta(df['ts'])
        result.drop(['date'],axis=1,inplace=True)
    #print('append:',thisdate,len(df),len(histbookdata),histbookdata['date'].unique())
    return result

def getDataFileName(equity,datestr,local):
    if local:
        return localroot + equity + '//' + equity + '-' + datestr + '.csv.gz'
    else:
        return s3rootraw + equity+'//'+equity+'-' + datestr + '.csv.gz'

def calcImb(df):
    df['buytot'] = df['buytot'] * 100
    df['selltot'] = df['selltot'] * 100
    df['vwap_other'] = df['otheramt'] / df['other']
    df['vwap_trf'] = df['trfamt'] / df['trf']
    df['mid_last'] = df['mid'].shift()
    df['vwapspread_other'] = df['vwap_other'] - df['mid_last']
    df['vwapspread_trf'] = df['vwap_trf'] - df['mid_last']

    df['buyother'] = df.apply(
        lambda r: r['other'] if (r['other'] > 0 and r['vwap_other'] > r['mid_last'] + spreadthreshold) \
            else 0, axis=1)
    df['sellother'] = df.apply(
        lambda r: r['other'] if (r['other'] > 0 and r['vwap_other'] < r['mid_last'] - spreadthreshold) \
            else 0, axis=1)
    df['unknownother'] = df['other'] - df['buyother'] - df['sellother']

    df['otherimb'] = df['buyother'] - df['sellother']

    df['buytrf'] = df.apply(lambda r: r['trf'] if (r['trf'] > 0 and r['vwap_trf'] > r['mid_last'] + spreadthreshold) \
        else 0, axis=1)
    df['selltrf'] = df.apply(lambda r: r['trf'] if (r['trf'] > 0 and r['vwap_trf'] < r['mid_last'] - spreadthreshold) \
        else 0, axis=1)
    df['unknowntrf'] = df['trf'] - df['buytrf'] - df['selltrf']
    df['trfimb'] = df['buytrf'] - df['selltrf']
    df['trfcntimb'] = df.apply(lambda r: 1 if r['trfimb']>0 else (-1 if r['trfimb']<0 else 0), axis=1)
    df['tradeimb'] = df['buy'] - df['sell']
    df['totimb'] = df['buytot'] - df['selltot']


def getDataAndFilter(gzfile):
    df = pd.read_csv(gzfile, compression='gzip',
                     names=datacolume).set_index('ts')
    df=df[df['bidpx']>0]
    df=df[df['askpx']>0]
    if len(df) < 1: return df
    df['mid'] = (df['bidpx'] + df['askpx']) / 2
    df['futuretrd'] = df['buy'] + df['sell'] +df['other']+df['trf']
    df['futuretrd']=df['futuretrd'].rolling(filterChgMin*60).sum().shift(-filterChgMin*60)
    firstTradeIndex = df[df['futuretrd'] > 0].index[0]
    df=df[firstTradeIndex:]
    if len(df[df['futuretrd']==0])>0:
        firstNoTradeIndex=df[df['futuretrd']==0].index[0]
        print('excl data for no trade',gzfile,len(df),len(df[:firstNoTradeIndex]))
        df=df[:firstNoTradeIndex]
    return df.drop(['futuretrd'],axis=1)

def getData(equity, datestr,local=True):
    #df = pd.read_csv(folder + equity + '-' + datestr + '.csv', names=datacolume).set_index('ts')
    df = getDataAndFilter(getDataFileName(equity,datestr,local))
    if len(df)<1:return df
    calcImb(df)
    df['tradeimb_cum'] = df['tradeimb'].cumsum()
    df['totimb_cum'] = df['totimb'].cumsum()
    df['otherimb_cum'] = df['otherimb'].cumsum()
    df['trfimb_cum'] = df['trfimb'].cumsum()
    df['alltradeimb_cum'] = df['tradeimb_cum'] + df['otherimb_cum']
    for side in ['_ask','_bid']:
        df['all'+side] = df[[x + side for x in venue]].sum(axis=1)
        for major in ['NYSE','NASDAQ','BATS']:
            df[major+'all' + side] = df[[x + side for x in venue if major in x]].sum(axis=1)
        df = df.drop([x + side for x in venue], axis=1)

    df.index = pd.TimedeltaIndex(df.index)
    return df


filterChgMin=10
filterChgThresh=0.015
def loadFeatureData(equity,local=True):
    df=pd.DataFrame()
    featurefolder=(localroot if local else s3root)+'features/'+equity+'/'
    fileiterables=([featurefolder+x for x in os.listdir(featurefolder)] if local else ['s3://'+x for x in fs.ls(featurefolder)])
    for file in fileiterables:
        thisdf=pd.DataFrame.from_csv(file)
        if len(thisdf) < 1: continue
        #if os.path.exists(bookfeaturefolder+equity+'//'+file):
        #    bookfeature = pd.DataFrame.from_csv(bookfeaturefolder+equity+'//'+file)
        #    if len(bookfeature)==0:continue
        #else: continue
        #thisdf=thisdf.merge(bookfeature.drop(['date','mid'],axis=1),left_index=True,right_index=True)
        #filter out the rest of the day if large move already occur
        thisdf['chg']=(thisdf['mid'].diff()/thisdf.iloc[0]['mid']).rolling(filterChgMin).sum()
        largeMove=thisdf[np.abs(thisdf['chg'])>filterChgThresh]
        if len(largeMove)>0:
            print(file,len(thisdf))
            print(largeMove[['mid','chg']])
            firstMover=largeMove.index[0]
            print('after filter:',len(thisdf[:firstMover]))
            thisdf=thisdf[:firstMover]
        #filter out datapoint if total trading size is 0 for the next n minutes
        df=pd.concat([df,thisdf.drop(['chg','running_max','running_min'],axis=1)])
    if len(df)>0:
        df['maxchg_abs']=np.abs(df[['maxup','maxdown']]).max(axis=1)
        df['maxchg']=np.sign(df['maxup']+df['maxdown'])*df['maxchg_abs']
        df['maxrange']=df['maxup']-df['maxdown']
        df['mindepth'] = df[['all_bid', 'all_ask']].min(axis=1)
        df.drop([venue + 'all_bid' for venue in ['NYSE', 'NASDAQ', 'BATS'] ],axis=1,inplace=True)
        df.drop([venue + 'all_ask' for venue in ['NYSE', 'NASDAQ', 'BATS']], axis=1, inplace=True)
    return df