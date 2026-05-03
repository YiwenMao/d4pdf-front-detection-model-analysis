import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
# build date list

#------------------------------------------------------------
#This is a script to calculate the averaged harmonic fitting of vorticity or MFC and plot them as in Figure S2-S3
#------------------------------------------------------------

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
dfm6=-1*pd.read_csv(fn,index_col=0)

fn = cH+'bft_'+tt+'_m7_mean.csv'
dfm7=-1*pd.read_csv(fn,index_col=0)

fn = cH+'bft_'+tt+'_m8_mean.csv'
dfm8=-1*pd.read_csv(fn,index_col=0)

'''
fig,axsC = plt.subplots(3,2,figsize=(12,8))
axsC=axsC.flatten()
for axs,sst in zip(axsC,['CC','GF','HA','MI','MP','MR']):

    for arr,cs,lb in zip([dfm6[sst],dfm7[sst],dfm8[sst]],['b','r','g'],['Jun','Jul','Aug']): 
        axs.hist(arr,edgecolor=[0.5,0.5,0.5],alpha=0.5,label=lb)
        axs.axvline(x=np.nanmean(arr),ls='-',color=cs)
        axs.axvline(x=np.percentile(arr,2.5),ls=':',color=cs)
        axs.axvline(x=np.percentile(arr,97.5),ls=':',color=cs)
    
        #axs.hist(dfm6[sst],edgecolor=[0.5,0.5,0.5],alpha=0.5,label='Jun')
        #axs.axvline(x=np.nanmean(dfm6[sst]),color='b')
        #arr = dfm6[sst].values
        #axs.axvline
        #axs.hist(dfm7[sst],edgecolor=[0.5,0.5,0.5],alpha=0.5,label='Jul')
        #axs.axvline(x=np.nanmean(dfm7[sst]),color='r')    
        #axs.hist(dfm8[sst],edgecolor=[0.5,0.5,0.5],alpha=0.5,label='Aug')
        #axs.axvline(x=np.nanmean(dfm8[sst]),color='g')
              
    axs.set_title('HFB_'+tt+'_'+sst+'-HPB')
    axs.set_xlabel('days')
    axs.legend()
plt.subplots_adjust(hspace=0.4)
#plt.subplots_adjust(wspace=0.3)
'''

#-------------------------------------------------------------
vnm='dmoist'
ynm='anomlaies (L+S)'
cH='/Users/yiwenmao/Documents/StaFront2/biau/'
dJJA_yts3_statC=dict()
dJJA_yts4_statC=dict()
for tt in ['2K','4K']:
    for snm in ['CC','GF','HA','MI','MP','MR']:
        fn = cH+'test_data/HFB_'+tt+'/'+snm+'_HarmonicData_'+vnm+'.mat'
        MC = sio.loadmat(fn)
        tmx = MC['dJJA_yts3']
        nt=len(tmx)
        arr = np.zeros((nt,3))
        arr[:,0]=np.nanmin(tmx,axis=1)
        arr[:,1]=np.nanmean(tmx,axis=1)
        arr[:,2]=np.nanmax(tmx,axis=1)
        dJJA_yts3_statC[tt+'_'+snm] = arr

        tmx = MC['dJJA_yts4']
        nt=len(tmx)
        arr = np.zeros((nt,3))
        arr[:,0]=np.nanmin(tmx,axis=1)
        arr[:,1]=np.nanmean(tmx,axis=1)
        arr[:,2]=np.nanmax(tmx,axis=1)
        dJJA_yts4_statC[tt+'_'+snm] = arr

fn = '/Users/yiwenmao/Documents/StaFront2/biau/'+'test_data/HPB/HarmonicData_'+vnm+'.mat'
MC = sio.loadmat(fn)
tmx = MC['dJJA_yts3']
arr = np.zeros((nt,3))
arr[:,0]=np.nanmin(tmx,axis=1)
arr[:,1]=np.nanmean(tmx,axis=1)
arr[:,2]=np.nanmax(tmx,axis=1)
dJJA_yts3_statC['HPB']=arr

tmx = MC['dJJA_yts4']
arr = np.zeros((nt,3))
arr[:,0]=np.nanmin(tmx,axis=1)
arr[:,1]=np.nanmean(tmx,axis=1)
arr[:,2]=np.nanmax(tmx,axis=1)
dJJA_yts4_statC['HPB']=arr

outH='/Users/yiwenmao/Documents/StaFront2/biau/test_data/'
fn = outH+'ObsBiau_'+vnm+'_ref.mat'
MC = sio.loadmat(fn)
dtL = MC['dtL']
JJA_ft = MC['JJA_ft'].flatten()
yts4_obs = MC['yts4'].flatten()
JJA_yts4_obs = MC['JJA_yts4'].flatten()
JJA_yts3_obs = MC['JJA_yts3'].flatten()

fig,axsC = plt.subplots(3,2,figsize=(12,8))
axsC=axsC.flatten()
for axs,sst in zip(axsC,['CC','GF','HA','MI','MP','MR']):
    axs.fill_between(dates,dJJA_yts3_statC['2K'+'_'+sst][:,0],dJJA_yts3_statC['2K'+'_'+sst][:,2],color='b',label='2K_'+sst,alpha=0.5)
    axs.fill_between(dates,dJJA_yts3_statC['4K'+'_'+sst][:,0],dJJA_yts3_statC['4K'+'_'+sst][:,2],color='g',label='4K_'+sst,alpha=0.5)
    axs.fill_between(dates,dJJA_yts3_statC['HPB'][:,0],dJJA_yts3_statC['HPB'][:,2],color='pink',label='HPB',alpha=0.5)
    #axs.plot(dates,dJJA_yts3_statC['4K'+'_'+sst],'g-',label='4K_'+sst)    

    axs.plot(dates,JJA_yts3_obs,'k-',label='ERA5')
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axs.legend()    
    axs.set_title('Harmonic Fit of MFC(L+S)')
    #axs.set_title('Harmonic Fit of vort(L+S)',fontsize=14)
    axs.axhline(y=0,color='k',linestyle='--')
    axs.tick_params(axis='x',labelsize=13)
    axs.tick_params(axis='y',labelsize=13)
plt.subplots_adjust(hspace=0.3)
