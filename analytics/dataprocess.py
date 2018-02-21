import numpy as np
import pandas as pd
import s3fs
import boto3
import os,io,gzip

import sys
if sys.version_info < (3, 0):
    import StringIO as myio
else:
    import io as myio



import multiprocessing as mp
import static


fs = s3fs.S3FileSystem()
s3_resource = boto3.resource('s3')

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
           'rolling_window':60*5,# seconds
           'autocorr_bin':10,
           'autocorr_window':30,
           'sample_interval':60,
           'bookrank_in_day':5
            }
####
def computeFeatureForEquities(to_run,startdate,enddate,local,parallel=True,output_local=None,output_verbose=False,overwrite=False):
    if parallel:
        processes = [
            mp.Process(target=computeFeatureForEquity,\
                   args=(equity,parameter, startdate,enddate,local,output_local,output_verbose,False,overwrite))\
            for equity in to_run]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
            print(p.pid,'done')
    else:
        for equity in to_run:
            computeFeatureForEquity(equity,parameter, startdate,enddate,local,output_local,output_verbose,False,overwrite)

def computeFeatureForEquitiesWraper(num,startdate,enddate,local,parallel=True,output_local=None,output_verbose=False,overwrite=False):
    if type(num)==int:
        to_run=[name[0] for name in static.equities_test if (name[0] not in static.equities_done)]
        if len(to_run)>num:
            to_run=to_run[:num]
        print('number',to_run)
    elif type(num)==str:
        to_run = [name[0] for name in static.equities_test if (name[0].startswith(num))]
        print('first letter',num,to_run)


    if len(to_run)<1: return
    computeFeatureForEquities(to_run, startdate, enddate, local, parallel=parallel, output_local=output_local, output_verbose=output_verbose,overwrite=overwrite)


def computeFeatureForEquity(equity, param, startdate,enddate,local=True,output_local=None,output_verbose=False,\
                            computeNewFeature=False,overwrite=False):
    #hdf = pd.HDFStore(folder+'features//'+equity+'.h5')
    #output=pd.DataFrame()
    #df = pd.DataFrame(columns=['date', 'range', 'change']).set_index('date')
    histbookdata =pd.DataFrame(columns=['date','spread','litvolume','volumebydepth','volumebymindepth']+[site + 'all_bid' for site in['','NYSE', 'NASDAQ', 'BATS']]+\
                                       [site + 'all_ask' for site in['','NYSE', 'NASDAQ', 'BATS']])
    datelist=[]
    if output_local==None: output_local=local
    #if computeSpread: return #save the template for adding features
    alldata=pd.DataFrame()

    if output_local:
        if computeNewFeature:
            outputpath = localroot+'new'
        else:
            outputpath = localroot
        if not os.path.exists(outputpath + 'features//' + equity + '//'):
            os.mkdir(outputpath + 'features//' + equity + '//')
    else:
        if computeNewFeature:
            outputpath = s3root+'new'
        else:
            outputpath = s3root
        #if not fs.exists(outputpath + 'features//' + equity + '//'):
        #    fs.mkdir(outputpath + 'features//' + equity + '//')
    if fs.exists(outputpath + 'features/' + equity + '_all.csv.gz'):
        if (overwrite):
            fs.rm(outputpath + 'features/' + equity + '_all.csv.gz')
        else:
            return
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
    datelist=sorted(list(set(datelist)))
    print(equity,len(datelist),datelist[0],datelist[-1])

    for thisdate in datelist:
        if thisdate<startdate or thisdate>enddate:
            print(thisdate,' skipped')
            continue
        file=equity+'-'+thisdate.strftime('%Y-%m-%d')+'.csv.gz'

        df = getDataAndFilter(prefixpath + '/' + file).reset_index()
        if len(df) > 0:
            df['date'] = thisdate.date()
            df['spread']=(df['askpx']-df['bidpx']).rolling(window=parameter['sample_interval']).mean()
            df['litvolume']=(df['buy']+df['sell']+df['sell']).rolling(window=parameter['sample_interval']).mean()
            for side in ['_ask', '_bid']:
                df['all' + side] = df[[x + side for x in venue]].sum(axis=1)
                for major in ['NYSE', 'NASDAQ', 'BATS']:
                    df[major + 'all' + side] = df[[x + side for x in venue if major in x]].sum(axis=1)
                df = df.drop([x + side for x in venue], axis=1)
            df['volumebydepth']=df['litvolume']/(df[[x + 'all_bid' for x in ['NYSE', 'NASDAQ', 'BATS']]].sum(axis=1)+
                                                 df[[x + 'all_ask' for x in ['NYSE', 'NASDAQ', 'BATS']]].sum(axis=1)+0.1)
            df['volumebymindepth'] = df['litvolume'] / (0.1+np.minimum(
            df[[x + 'all_bid' for x in ['NYSE', 'NASDAQ', 'BATS']]].sum(axis=1),
            df[[x + 'all_ask' for x in ['NYSE', 'NASDAQ', 'BATS']]].sum(axis=1)))

            outputfile=outputpath + 'features/' + equity + '/' + thisdate.strftime('%Y%m%d') + '.csv'

            if output_local:
                if not os.path.exists(outputfile):
                    if computeNewFeature:
                        #computeFeaturesNew(df, thisdate, param, histbookdata,['litvolume']).to_csv(outputfile)
                        #computeFeaturesNew(df, thisdate, param, histbookdata, ['spread']).to_csv(outputfile)
                        computeFeaturesNew(df, thisdate, param, histbookdata, ['volumebydepth','volumebymindepth']).to_csv(outputfile)
                    else:
                        computeFeatures(df, thisdate, param, histbookdata).to_csv(outputfile)
            else:
                #print('check exist',outputfile,fs.exists(outputfile))
                if not fs.exists(outputfile):
                    print(thisdate, len(histbookdata), prefixpath + '/' + file)
                    result=computeFeatures(df, thisdate, param, histbookdata)
                    alldata = pd.concat([alldata, result])
                    if output_verbose:
                        bytes_to_write = result.to_csv(None).encode()
                        with fs.open(outputfile, 'wb') as f:
                            f.write(bytes_to_write)
                else:
                    result=pd.read_csv(outputfile,index_col=0)
                    alldata = pd.concat([alldata, result])
            histbookdata=updatehistbookdata(df,param,histbookdata)
    #write the all data
    if computeNewFeature: return #don't need to combine for a patch run
    if output_local:
        alldata.to_csv(outputpath + 'features//' + equity + '.csv.gz',compression='gzip')
    else:
        csv_buffer = myio.StringIO()
        alldata.to_csv(csv_buffer)
        print(len(alldata))
        #csv_buffer.seek(0)
        gz_buffer=io.BytesIO()
        with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
            #gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))
            gz_file.write(csv_buffer.getvalue().encode('utf-8'))

        s3_object = s3_resource.Object('equity-flash', 'features/' + equity + '_all.csv.gz')
        s3_object.put(Body=gz_buffer.getvalue())

def updatehistbookdata(df,param,histbookdata):
    if len(histbookdata['date'].unique())>=param['bookrank_in_day']:
        mindate=sorted(histbookdata['date'].unique())[0]
        histbookdata= histbookdata[histbookdata.date>mindate]
        #print(thisdate, 'drop ' + str(mindate),len(histbookdata))
    #update histbookdata

    return pd.concat([histbookdata,df[['date','spread','litvolume','volumebydepth','volumebymindepth']+[site + 'all_bid' for site in['','NYSE', 'NASDAQ', 'BATS']]+\
                                             [site + 'all_ask' for site in['','NYSE', 'NASDAQ', 'BATS']]]])

def computeFeatures(df,thisdate,param,histbookdata):
    bookfeatures = computeBookFeature(df,thisdate,param,histbookdata)
    spreadfeature = computeSpreadFeature(df, thisdate, param, histbookdata)
    volumefeature = computeGeneralFeature(df, thisdate, param, histbookdata,['litvolume'])
    if (len(bookfeatures)<1) or (len(spreadfeature)<1):
        return pd.DataFrame()
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
    result = result.merge(spreadfeature, how='left', left_index=True, right_index=True)
    result = result.merge(volumefeature, how='left', left_index=True, right_index=True)

    result=result.dropna().resample(str(param['sample_interval'])+'S').last()
    return result

def computeFeaturesNew(df,thisdate,param,histbookdata,featurenames):
    newFeature = computeGeneralFeature(df,thisdate,param,histbookdata,featurenames)
    if len(newFeature)<1: return pd.DataFrame()
    result = newFeature

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
def computeSpreadFeature(df,thisdate,param,histbookdata):
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
        for key in ['spread']:
            interprank(key)

        result['date']= thisdate
        result.index = pd.to_datetime(result['date']) + pd.to_timedelta(df['ts'])
        result.drop(['date'],axis=1,inplace=True)
    #print('append:',thisdate,len(df),len(histbookdata),histbookdata['date'].unique())
    return result
def computeGeneralFeature(df,thisdate,param,histbookdata,featurenames):
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
        for key in featurenames:
            interprank(key)

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
    errorcnt=0
    while errorcnt<3:
        try:
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
            return df.drop(['futuretrd'], axis=1)
        except:
            print("error:", sys.exc_info()[0])
            errorcnt+=1
    raise RuntimeError("get data fail: "+gzfile)

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
filterChgThresh=0.02

def loadFeatureDataBulk(equity,featurecols,truncPostMove=True,temperAtOpen=[],resample=1,startdate=pd.datetime(1979,9,26),enddate=pd.datetime(2020,2,2)):
    if not fs.exists(s3root+'features/'+equity+'_all.csv.gz'):
        print('cannot find file: ', equity)
        return []
    df=pd.read_csv(s3root+'features/'+equity+'_all.csv.gz',compression='gzip',index_col=0)
    #check data
    x = df.groupby(level=0)['litvolume'].count()
    if len(x[x > 1]) > 0:
        print('duplicate time index: ',equity)
        return []
    #if 'Unnamed: 0.1' in df.columns:
    #    print(' index column missing : ' + s3root + 'features/' + equity + '_all.csv.gz')
    #    df.set_index('Unnamed: 0.1', inplace=True)
    df.index = pd.to_datetime(df.index)
    filteredDF=pd.DataFrame()
    datelist=df['date'].unique()
    #print(equity, len(datelist),datelist)
    #print(df.groupby('date')['mid'].count()[:20])

    for date in datelist:
        if (pd.to_datetime(date)<startdate) or (pd.to_datetime(date)>enddate):continue
        thisdf=df[df['date']==date].copy()
        if len(thisdf)<filterChgMin:
            #print(equity,date,len(thisdf))
            continue
        firsttime=thisdf.index[0]
        thisdf['firsttime']=thisdf.index-firsttime
        #if os.path.exists(bookfeaturefolder+equity+'//'+file):
        #    bookfeature = pd.DataFrame.from_csv(bookfeaturefolder+equity+'//'+file)
        #    if len(bookfeature)==0:continue
        #else: continue
        #thisdf=thisdf.merge(bookfeature.drop(['date','mid'],axis=1),left_index=True,right_index=True)
        #filter out the rest of the day if large move already occur
        thisdf['chg']=(thisdf['mid'].diff()/thisdf.iloc[0]['mid']).rolling(filterChgMin).sum()
        largeMove=thisdf[np.abs(thisdf['chg'])>filterChgThresh]
        if truncPostMove and (len(largeMove)>0):
            #print(equity,date,len(thisdf))
            #print(largeMove[['mid','chg']])
            firstMover=largeMove.index[0]
            #print('after filter:',len(thisdf[:firstMover]))
            thisdf=thisdf[:firstMover]
        #filter out datapoint if total trading size is 0 for the next n minutes
        if len(temperAtOpen)>0:
            thisdf['factor']=range(len(thisdf))
            thisdf['factor']= np.tanh(thisdf['factor'])

#factor is from 0 to 1, in order to neutralize the first few minutes
        thisdf['mindepth'] = thisdf[['all_bid', 'all_ask']].min(axis=1)
        for venue in ['NYSE', 'NASDAQ', 'BATS']:
            thisdf['mindepth'+venue] = thisdf[[venue+'all_bid', venue+'all_ask']].min(axis=1)
            thisdf['mindepth'+venue] = 2 * (thisdf['mindepth'+venue] - 0.5)
        for feature in ['mindepth','litvolume','spread']:
            thisdf[feature] =2*(thisdf[feature]-0.5)
            if feature in temperAtOpen:
                thisdf[feature]=thisdf[feature]*thisdf['factor']

        thisdf.drop([venue + 'all_bid' for venue in ['','NYSE', 'NASDAQ', 'BATS'] ],axis=1,inplace=True)
        thisdf.drop([venue + 'all_ask' for venue in ['','NYSE', 'NASDAQ', 'BATS']], axis=1, inplace=True)

        for feature in featurecols:
            thisdf[feature + '_1'] = thisdf[feature].shift(2)
            thisdf[feature + '_2'] = thisdf[feature].shift(4)

        thisdf.dropna(inplace=True)

        filteredDF=pd.concat([filteredDF,thisdf.iloc[::resample].drop(['chg','running_max','running_min'],axis=1)])
    if len(filteredDF)>0:
        filteredDF['maxchg_abs']=np.abs(filteredDF[['maxup','maxdown']]).max(axis=1)
        filteredDF['maxchg']=np.sign(filteredDF['maxup']+filteredDF['maxdown'])*filteredDF['maxchg_abs']
        filteredDF['maxrange']=filteredDF['maxup']-filteredDF['maxdown']

    return filteredDF
