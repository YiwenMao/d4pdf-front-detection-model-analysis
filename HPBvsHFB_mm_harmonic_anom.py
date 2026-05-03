#figure(4e-4h)
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

#===================================================
#a script to calculate the multi model mean of harmonic fitting and anomalies of vorticity and MFC for Fig4(e,f,g,h)
#===================================================


# build date list
start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)
dates=np.array(dates)

#-------------------------------------------------------------
vnm='vort'
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

dMM3=dict()
for tt in ['2K','4K']:
    cmx=np.zeros((92,6))
    for sst,k in zip(['CC','GF','HA','MI','MP','MR'],[0,1,2,3,4,5]):
        cmx[:,k]=dJJA_yts3_statC[tt+'_'+sst][:,1]
    mm = np.zeros((92,3))
    mm[:,0] = np.nanmin(cmx,axis=1)
    mm[:,1] = np.nanmean(cmx,axis=1)
    mm[:,2] = np.nanmax(cmx,axis=1)
    dMM3[tt]=mm

dMM4=dict()
for tt in ['2K','4K']:
    cmx=np.zeros((92,6))
    for sst,k in zip(['CC','GF','HA','MI','MP','MR'],[0,1,2,3,4,5]):
        cmx[:,k]=dJJA_yts4_statC[tt+'_'+sst][:,1]
    mm = np.zeros((92,3))
    mm[:,0] = np.nanmin(cmx,axis=1)
    mm[:,1] = np.nanmean(cmx,axis=1)
    mm[:,2] = np.nanmax(cmx,axis=1)
    dMM4[tt]=mm

fig,axsC = plt.subplots(2,1,figsize=(6,8))
axsC = axsC.flatten()
axs = axsC[0]
mm = dMM3['2K']
axs.fill_between(dates,mm[:,0],mm[:,2],color='b',alpha=0.4,label='HFB 2K (MultiM)')
axs.plot(dates,mm[:,1],color='b',lw=2)
mm = dMM3['4K']
axs.fill_between(dates,mm[:,0],mm[:,2],color='g',alpha=0.4,label='HFB 4K (MultiM)')
axs.plot(dates,mm[:,1],color='g',lw=2)
axs.fill_between(dates,dJJA_yts3_statC['HPB'][:,0],dJJA_yts3_statC['HPB'][:,2],color='pink',label='HPB',alpha=0.5)
axs.plot(dates,dJJA_yts3_statC['HPB'][:,1],'m-')    
axs.plot(dates,JJA_yts3_obs,'k:',label='Obs')
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
axs.legend()  
axs.set_xlabel('dates')
axs.set_title('Normalized harmonic Fit of Vort(L+S) between 30N-40N')

axs = axsC[1]
mm = dMM4['2K']
axs.fill_between(dates,mm[:,0],mm[:,2],color='b',alpha=0.4,label='HFB 2K (MultiM)')
axs.plot(dates,mm[:,1],color='b',lw=2)
mm = dMM4['4K']
axs.fill_between(dates,mm[:,0],mm[:,2],color='g',alpha=0.4,label='HFB 4K (MultiM)')
axs.plot(dates,mm[:,1],color='g',lw=2)
axs.fill_between(dates,dJJA_yts4_statC['HPB'][:,0],dJJA_yts4_statC['HPB'][:,2],color='pink',label='HPB',alpha=0.5)
axs.plot(dates,dJJA_yts4_statC['HPB'][:,1],'m-')    
axs.plot(dates,JJA_yts4_obs,'k:',label='Obs')
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
axs.legend()  
axs.set_xlabel('dates')
axs.set_title('Normalized Anomalies of Vort(L+S) between 30N-40N')

plt.subplots_adjust(hspace=0.3)