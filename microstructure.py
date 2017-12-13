import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

periods_lookback=5
agent_n=100
#agent_n=1
agent_pov=0.03
agent_size_min=100
agent_size_max=200
shocksize=0
#shocksize=200

total_period=300
noise_size_min=20
noise_size_max=60

random_seed=1

total_size=(noise_size_min+noise_size_max)/2*total_period +agent_n*(agent_size_min+agent_size_max)/2+shocksize
const_rate=total_size/total_period*agent_pov

class povAlgo:
    def __init__(self, total_size, pov, start_time,dir):
        self.size = total_size
        self.dir=dir
        self.pov=pov
        self.rate =pov
        self.start=start_time
        self.accumsize=0

    #def launch(self,time):
    #    self.accumsize=0
    #    self.launched = True
    def updatesize(self,lookbackvol_rate):
        if self.accumsize>=self.size:
            current=0
        else:
            current=min(self.size-self.accumsize,round(self.pov*lookbackvol_rate))
        self.accumsize+=current
        return current
class constAlgo:
    def __init__(self, total_size, rate, start_time,dir):
        self.size = total_size
        self.dir = dir
        self.rate=rate
        self.start=start_time
        self.accumsize=0
    #def launch(self,time):
    #    self.accumsize=0
    #    self.launched = True
    def updatesize(self,lookback_rate=0):
        if self.accumsize>=self.size:
            current=0
        else:
            current=min(self.size-self.accumsize,self.rate)
        self.accumsize+=current
        return current

def getPOVAgents(agent_n,agent_pov,agent_size_min,agent_size_max,time_start_min,time_start_max):
    np.random.seed(random_seed)
    agents = [povAlgo(size, agent_pov, start,direction) for \
              size, start,direction in zip(np.random.randint(agent_size_min, agent_size_max, agent_n), \
                                 np.random.randint(time_start_min, time_start_max, agent_n), \
                                           [1]*int(agent_n/2)+[-1]*int(agent_n/2))]
    return agents

def getConstAgents(agent_n,const_rate,agent_size_min,agent_size_max,time_start_min,time_start_max):
    np.random.seed(random_seed)
    agents = [constAlgo(size, const_rate, start,direction) for \
              size, start,direction in zip(np.random.randint(agent_size_min, agent_size_max, agent_n), \
                                 np.random.randint(time_start_min, time_start_max, agent_n), \
                                           [1] * int(agent_n / 2) + [-1] * int(agent_n / 2))]
    return agents



def runSim(agents,noise_size_min,noise_size_max,total_period,shock_time=0,shock_size=0):
    np.random.seed(random_seed)

    #algo and non-algo are signed, algo_volume is sum of abs(algo size)
    marketDF=pd.DataFrame({'non-algo':np.random.randint(noise_size_min,noise_size_max,total_period)*np.sign(np.random.random(total_period)-0.5),\
                       'shock':0,'algo':0,'particpation':0,'signedpart':0,'algo_volume':0},index=range(total_period))

    marketDF['shock']=0
    marketDF.loc[shock_time,'shock']=shock_size
    marketDF['volume']=np.abs(marketDF[['non-algo','algo','shock']]).sum(axis=1)


    for i in range(total_period):
        agent_volume=0
        agent_size=0
        participation = 0
        signedpart=0
        if i<periods_lookback:
            continue
        lookback_rate=marketDF[i-periods_lookback:i]['volume'].mean()
        for agent in agents:
            if i>=agent.start:
                thisvolume=agent.updatesize(lookback_rate)
                agent_volume+=thisvolume
                agent_size+=thisvolume*agent.dir
                if agent.accumsize < agent.size:
                    participation+=agent.rate #rate has differenet units in POV or const agent
                    signedpart+=agent.rate*agent.dir
        marketDF.loc[i,'algo']=agent_size
        marketDF.loc[i,'participation']=participation
        marketDF.loc[i,'signedpart']=signedpart
        marketDF.loc[i,'algo_volume']=agent_volume
        marketDF.loc[i,'volume']=abs(marketDF.loc[i,'non-algo'])+abs(marketDF.loc[i,'shock'])+marketDF.loc[i,'algo_volume']
    return marketDF

agents_pov = getPOVAgents(agent_n,agent_pov,agent_size_min,agent_size_max,10,150)
agents_const = getConstAgents(agent_n,const_rate,agent_size_min,agent_size_max,10,150)
marketDF_POV = runSim(agents_pov,noise_size_min,noise_size_max,total_period,shock_time=0,shock_size=0)
marketDF_Const = runSim(agents_const,noise_size_min,noise_size_max,total_period,shock_time=0,shock_size=0)

print(marketDF_POV['algo_volume'].sum()/marketDF_POV['volume'].sum(),marketDF_Const['algo_volume'].sum()/marketDF_Const['volume'].sum())


fig, axes = plt.subplots(2, 2, figsize=(10, 6))
plt.suptitle('Simulation: POV executions generate spiky volume profile',fontsize=16)
#plt.text(-50, -12, 'agent no='+str(agent_n)+\
#         ', pov='+str(int(agent_pov*100))+\
#         '%, execution total pct='+str(int(execution_volume_ratio*100))+\
#         '%'
#         ,fontsize=12,style='italic')

pd.DataFrame({'algo':marketDF_POV['algo_volume'],
             'non-algo':np.abs(marketDF_POV['non-algo']),
              'participation':marketDF_POV['participation']}).plot(ax=axes[0][0],title='POV',secondary_y=['participation'],color=['b','g','0.65'])
ymin, ymax = axes[0][0].get_ylim()
pd.DataFrame({'algo':marketDF_Const['algo_volume'],
              'non-algo':np.abs(marketDF_Const['non-algo'])}).plot(ax=axes[1][0],title='Const Rate - TWAP',color=['b','g'])
axes[1][0].set_ylim(ymin,ymax)
marketDF_POV['volume'].hist(ax=axes[0][1])
marketDF_Const['volume'].hist(ax=axes[1][1])
#plt.legend(loc='upper right')
plt.show()


#agents_const_success=np.array([agent.accumsize>=agent.size for (agent_pov,agent_const) in agents])

#povDF=pd.DataFrame({'agent_pov':marketDF['agent_pov']/marketDF['volume_pov'],'shock':marketDF['shock']/marketDF['volume'], \
#                    'noise': marketDF['noise'] / marketDF['volume']})

fig_imb, axes_imb = plt.subplots(2, 1, figsize=(6, 8))
marketDF_POV[['algo','non-algo']].sum(axis=1).plot(kind='bar',ax=axes_imb[0],title='Order Imbalance',fontsize=10)

ticks = axes_imb[0].xaxis.get_ticklocs()
ticklabels = [l.get_text() for l in axes_imb[0].xaxis.get_ticklabels()]
axes_imb[0].xaxis.set_ticks(ticks[::50])
axes_imb[0].xaxis.set_ticklabels(ticklabels[::50],rotation=90)
axes_imb[0].xaxis.set_visible(False)
fig_imb.tight_layout()

marketDF_POV[['signedpart']].plot(ax=axes_imb[1],legend=False,title='Signed Total Participation',fontsize=10)
fig_imb.show()