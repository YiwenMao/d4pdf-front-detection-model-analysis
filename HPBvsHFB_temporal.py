import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

#======================================================
#calculate the boostrap difference in average number of days with Baiu Fronts in June, July and August respectively per year for HFB 2K/4K (Fig.(a-c))
#======================================================
# build date list
start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)
dates=np.array(dates)

tt = '2K'
cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/'
fn = cH+'bft_'+tt+'_m6_mean.csv'
dfm6_2K=-1*pd.read_csv(fn,index_col=0)
dfm6_2K['mean']=dfm6_2K.mean(axis=1)

fn = cH+'bft_'+tt+'_m7_mean.csv'
dfm7_2K=-1*pd.read_csv(fn,index_col=0)
dfm7_2K['mean']=dfm7_2K.mean(axis=1)

fn = cH+'bft_'+tt+'_m8_mean.csv'
dfm8_2K=-1*pd.read_csv(fn,index_col=0)
dfm8_2K['mean']=dfm8_2K.mean(axis=1)

dfmA=dict()
dfmA[tt] = dfm6_2K+dfm7_2K+dfm8_2K

tt = '4K'
cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/'
fn = cH+'bft_'+tt+'_m6_mean.csv'
dfm6_4K=-1*pd.read_csv(fn,index_col=0)
dfm6_4K['mean']=dfm6_4K.mean(axis=1)

fn = cH+'bft_'+tt+'_m7_mean.csv'
dfm7_4K=-1*pd.read_csv(fn,index_col=0)
dfm7_4K['mean']=dfm7_4K.mean(axis=1)

fn = cH+'bft_'+tt+'_m8_mean.csv'
dfm8_4K=-1*pd.read_csv(fn,index_col=0)
dfm8_4K['mean']=dfm8_4K.mean(axis=1)
dfmA[tt] = dfm6_4K+dfm7_4K+dfm8_4K


snm = 'mean'
fig,axsC = plt.subplots(1,3,figsize=(15,3))
axsC=axsC.flatten()
axs = axsC[0]
for arr,cs,lb in zip([dfm6_2K[snm],dfm7_2K[snm],dfm8_2K[snm]],['b','r','g'],['Jun','Jul','Aug']):
    axs.hist(arr,edgecolor=[0.5,0.5,0.5],alpha=0.5,label=lb)
    axs.axvline(x=np.nanmean(arr),ls='-',color=cs)
    axs.axvline(x=np.percentile(arr,2.5),ls=':',color=cs)
    axs.axvline(x=np.percentile(arr,97.5),ls=':',color=cs)
axs.set_title('HFB_2K-HPB',fontsize=14)
axs.set_xlabel('days',fontsize=14)
axs.legend()

axs = axsC[1]
for arr,cs,lb in zip([dfm6_4K[snm],dfm7_4K[snm],dfm8_4K[snm]],['b','r','g'],['Jun','Jul','Aug']):
    axs.hist(arr,edgecolor=[0.5,0.5,0.5],alpha=0.5,label=lb)
    axs.axvline(x=np.nanmean(arr),ls='-',color=cs)
    axs.axvline(x=np.percentile(arr,2.5),ls=':',color=cs)
    axs.axvline(x=np.percentile(arr,97.5),ls=':',color=cs)
axs.set_title('HFB_4K-HPB',fontsize=14)
axs.set_xlabel('days',fontsize=14)
axs.legend()

axs = axsC[2]
arr = dfmA['2K'][snm].values
axs.hist(arr,edgecolor=[0.5,0.5,0.5],color='b',alpha=0.3,label='HFB 2K')
axs.axvline(x=np.nanmean(arr),ls='-',lw=2,color='b')
axs.axvline(x=np.percentile(arr,2.5),ls=':',color='b')
axs.axvline(x=np.percentile(arr,97.5),ls=':',color='b')

arr = dfmA['4K'][snm].values
axs.hist(arr,edgecolor=[0.5,0.5,0.5],color='g',alpha=0.3,label='HFB 4K')
axs.axvline(x=np.nanmean(arr),ls='-',lw=2,color='g')
axs.axvline(x=np.percentile(arr,2.5),ls=':',color='g')
axs.axvline(x=np.percentile(arr,97.5),ls=':',color='g')
axs.set_xlabel('days',fontsize=14)
axs.set_title('Avg Baiu Frontal days in JJA (HFB-HPB)',fontsize=14)
axs.legend()

for axs in axsC:
    axs.tick_params(axis='x',labelsize=13)
    axs.tick_params(axis='y',labelsize=13)

