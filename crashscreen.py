import numpy as np
import pandas as pd
import os
import s3fs


local=True

if not local:
    fs = s3fs.S3FileSystem()

###boto3 access
import boto3
#client=boto3.client('s3')
#resource=boto3.resource('s3')
#my_bucket=resource.Bucket('eastend-flash')
###boto3 access

####s3fs access
#with fs.open('s3://eastend-flash/flash/AAPL/AAPL-2017-09-12.csv.gz') as f:
#    df = pd.read_csv(f, compression='gzip')
#####s3fs access


venue=['NYSE_MKT','NASDAQ_BX','BATS_EDGA','BATS_EDGX','CHX','NYSE','NYSE_ARCA',\
       'NASDAQ','IEX','NASDAQ_PSX','BATS_BYX','BATS_BZX']
folder='Data//'
datacolume=['ts','bidpx','askpx','buy','sell','other','trf','buyamt','sellamt','otheramt','trfamt']\
            +[x+'_bid' for x in venue]+[x+'_ask' for x in venue]+['buytot','selltot']


def getHLPct(gzfile):
    #df = pd.read_csv(folder + equity + '//'+equity+ '-' + datestr + '.csv.gz', compression='gzip',names=datacolume).set_index('ts')
    df = pd.read_csv(gzfile, compression='gzip',names=datacolume).set_index('ts')
    df = df[df['bidpx'] > 0]
    df = df[df['askpx'] > 0]
    df['mid'] = (df['bidpx'] + df['askpx']) / 2
    last=df['mid'].iloc[-1]
    high=df['mid'].max()
    low=df['mid'].min()
    first=df['mid'].iloc[0]
    return [(high-low)/last,(last-first)/last]

equity='ZTS'
df=pd.DataFrame(columns=['date','range','change']).set_index('date')

if local:
    for file in os.listdir(folder+equity):
        filenameparser=file.split('.')[0].split('-')
        thisdate=pd.datetime(int(filenameparser[1]),int(filenameparser[2]),int(filenameparser[3]))
        df.loc[thisdate]=getHLPct(folder+equity+'//'+file)
else:
    for file in fs.ls('s3://eastend-flash/flash/'+equity+'//'):
        filenameparser = file.split('/')[-1].split('.')[0].split('-')
        thisdate = pd.datetime(int(filenameparser[1]), int(filenameparser[2]), int(filenameparser[3]))
        df.loc[thisdate] = getHLPct( 's3://' + file)



max_range=0.02
pd.cut(df[df['range']>max_range]['change'].values,np.arange(-max_range,max_range+0.01,0.01)).value_counts()

