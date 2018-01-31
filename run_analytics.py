import analytics.dataprocess as dataprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import static
import gzip
import io
equity = 'DLPH'

#####generate features
dataprocess.computeFeatureForEquity(equity,dataprocess.parameter,pd.datetime(2017,2,3),pd.datetime(2017,11,29),local=False,output_local=True)
#####generate features

#####tmp patch to add new feature
for equity in static.equities_done[21:]:
    temp = pd.read_csv('s3://equity-flash/features/'+equity+'_all.csv.gz', compression='gzip', index_col=0)
    x = temp.groupby(level=0)['litvolume'].count()
    if len(x[x>1])>0:
        print('run:',equity)
        dataprocess.computeFeatureForEquity(equity, dataprocess.parameter, pd.datetime(2017, 2, 3),
                                        pd.datetime(2017, 11, 29), local=False, overwrite=True)



#add hoc:
for equitiy in static.equities_done:
    dataprocess.computeFeatureForEquities(['ALGN','ALK'],pd.datetime(2017,2,3),pd.datetime(2017,11,29),False,overwrite=True)

#for equity in static.equities_done[1:]:
featurename='spread'
for equity in static.equities_done:
    localfolder='Data//newfeatures/' + equity + '/'
    #remotefolder='s3://equity-flash/features/'+equity+'/'
    remotefolder = 's3://equity-flash/features/'
    print(equity)
    df = pd.read_csv(remotefolder + equity+'_all.csv.gz',compression='gzip',index_col=0)
    if 'Unnamed: 0.1' in df.columns:
        df.set_index('Unnamed: 0.1', inplace=True)
    df.index = pd.to_datetime(df.index)
    if len(df)<1: continue
    alldata=pd.DataFrame()
    for file in os.listdir(localfolder):
        if file.startswith('.'): continue
        spreaddata= pd.DataFrame.from_csv(localfolder+file)
        if len(spreaddata)>0:
            alldata=pd.concat([alldata,spreaddata])

            #bytes_to_write = df.to_csv(None).encode()
            #print('write: '+ remotefolder+file)
            #with dataprocess.fs.open(remotefolder+file, 'wb') as f:
            #    f.write(df.to_csv(None).encode())
        #alldata=pd.concat([alldata,df])
    csv_buffer = io.StringIO()
    if featurename in df.columns:
        df.drop(featurename,axis=1,inplace=True)
    df.merge(alldata,left_index=True,right_index=True,how='left').to_csv(csv_buffer)
    csv_buffer.seek(0)
    gz_buffer=io.BytesIO()
    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        gz_file.write(bytes(csv_buffer.getvalue(), 'utf-8'))
    s3_object = dataprocess.s3_resource.Object('equity-flash', 'features/' + equity + '_all.csv.gz')
    s3_object.put(Body=gz_buffer.getvalue())
#####tmp patch

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
