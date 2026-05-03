import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
# build date list

#------------------------------------------------------------
#This is a script to calculate the averaged anomalies of vorticity or MFC and plot them as in Figure S4-S5
#------------------------------------------------------------
start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)
dates=np.array(dates)
#-------------------------------------------------------------
vnm='vort'#or 'dmoist' for MFC
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

fig,axsC = plt.subplots(3,2,figsize=(12,10))
axsC=axsC.flatten()
for axs,sst in zip(axsC,['CC','GF','HA','MI','MP','MR']):
    axs.fill_between(dates,dJJA_yts4_statC['2K'+'_'+sst][:,0],dJJA_yts4_statC['2K'+'_'+sst][:,2],color='b',label='2K_'+sst,alpha=0.4)
    axs.plot(dates,dJJA_yts4_statC['2K'+'_'+sst][:,1],color='b')

    axs.fill_between(dates,dJJA_yts4_statC['4K'+'_'+sst][:,0],dJJA_yts4_statC['4K'+'_'+sst][:,2],color='g',label='4K_'+sst,alpha=0.4)
    axs.plot(dates,dJJA_yts4_statC['4K'+'_'+sst][:,1],color='g')

    axs.fill_between(dates,dJJA_yts4_statC['HPB'][:,0],dJJA_yts4_statC['HPB'][:,2],color='pink',label='HPB',alpha=0.5)
    axs.plot(dates,dJJA_yts4_statC['HPB'][:,1],'r-')    

    axs.plot(dates,JJA_yts4_obs,'k-',label='ERA5')
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axs.legend()    
    #axs.set_title('Harmonic Fit of Cvg.Moist(L+S)')
    #axs.set_title('Anomalies of MFC(L+S)',fontsize=14)
    axs.set_title('Anomalies of vort(L+S)',fontsize=14)
    axs.axhline(y=0,color='k',linestyle='--')
    axs.tick_params(axis='x',labelsize=13)
    axs.tick_params(axis='y',labelsize=13)
plt.subplots_adjust(hspace=0.3)