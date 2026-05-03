import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import t as tdist
from scipy.stats import skew, kurtosis
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime, timedelta

#=================================================================================
#calculate relative frequency of Baiu fronts (defined as the ratio of the number of Baiu fronts that occurred at a date in JJA to the total number of years for prediction) calculated from HPB, and multimodel mean of HFB 2K and 4K scenarios
#=================================================================================
def getCI(sampleC):
    ref_mean = np.nanmean(sampleC,axis=1)
    ref_CIup = np.percentile(sampleC,97.5,axis=1)
    ref_CIdn = np.percentile(sampleC,2.5,axis=1)   
    return ref_mean, ref_CIup, ref_CIdn

start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)

Nssn = len(dates)
dssn = np.zeros((Nssn,2))
for di in np.arange(0,Nssn):
    dssn[di,0]=dates[di].month
    dssn[di,1]=dates[di].day
dates=np.array(dates)

CIm=dict()
CIup=dict()
CIdn=dict()
cH = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'
fn = cH+'bft_HPB_time.mat'
MC = sio.loadmat(fn)
tarrC_HPB = MC['data']
ref_mean, ref_CIup, ref_CIdn=getCI(tarrC_HPB)
CIm['HPB']=ref_mean
CIup['HPB']=ref_CIup
CIdn['HPB']=ref_CIdn

dtarrC = dict()
dCIm=dict()
dCIup=dict()
dCIdn=dict()
for tt in ['2K','4K']:
    for sst in ['CC','GF','HA','MI','MP','MR']:
        fn = cH+'bft_HFB_'+tt+'_'+sst+'_time.mat'
        MC = sio.loadmat(fn)
        tarrC = MC['data']
        ref_mean, ref_CIup, ref_CIdn=getCI(tarrC)
        CIm['HFB_'+tt+'_'+sst]=ref_mean
        CIup['HFB_'+tt+'_'+sst]=ref_CIup
        CIdn['HFB_'+tt+'_'+sst]=ref_CIdn

        dtarrC['HFB_'+tt+'_'+sst] = MC['data']-tarrC_HPB
        ref_mean, ref_CIup, ref_CIdn=getCI(dtarrC['HFB_'+tt+'_'+sst])
        dCIm['HFB_'+tt+'_'+sst]=ref_mean
        dCIup['HFB_'+tt+'_'+sst]=ref_CIup
        dCIdn['HFB_'+tt+'_'+sst]=ref_CIdn
        #break
    #break


dmeanC=dict()
dsigC=dict()
dmaxC=dict()
dminC=dict()
for tt in ['2K','4K']:
    meanC=np.zeros((92,6))
    sigC=np.zeros((92,))
    for sst,k in zip(['CC','GF','HA','MI','MP','MR'],[0,1,2,3,4,5]):
        meanC[:,k]=CIm['HFB_'+tt+'_'+sst]
        dsig = ~((dCIup['HFB_'+tt+'_'+sst]>0)&(dCIdn['HFB_'+tt+'_'+sst]<0))
        sigC=sigC+dsig
        
    maxC = np.max(meanC,axis=1)
    minC = np.min(meanC,axis=1)
    meanC = np.mean(meanC,axis=1)
    dmeanC[tt]=meanC
    dmaxC[tt]=maxC
    dminC[tt]=minC
    dsigC[tt]=sigC

fig,axs = plt.subplots(1,1,figsize=(6,4))

axs.fill_between(dates,CIup['HPB'],CIdn['HPB'],color='r',label='HPB',alpha=0.3)
axs.fill_between(dates,dmaxC['2K'],dminC['2K'],color='b',label='HFB_2K (MultiM)',alpha=0.3)
axs.fill_between(dates,dmaxC['4K'],dminC['4K'],color='g',label='HFB_4K (MultiM)',alpha=0.3)
axs.plot(dates,CIm['HPB'],ls='--',color='r',lw=1)
axs.plot(dates,dmeanC['2K'],ls='--',color='b',lw=1)
axs.plot(dates,dmeanC['4K'],ls='--',color='g',lw=1)
d2K = dsigC['2K']>=4
axs.plot(dates[d2K],dmeanC['2K'][d2K],'b.')
d2K = dsigC['2K']==6
axs.plot(dates[d2K],dmeanC['2K'][d2K],'bo',mfc='none')

d4K = dsigC['4K']>=4
axs.plot(dates[d4K],dmeanC['4K'][d4K],'g.')
d2K = dsigC['4K']==6
axs.plot(dates[d4K],dmeanC['4K'][d4K],'go',mfc='none')

axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
axs.tick_params(axis='x', labelsize=14)              # size for x ticks
axs.tick_params(axis='y', labelsize=14)              # size for y ticks
axs.legend()
axs.set_ylabel('Freq of Baiu Fronts',fontsize=14)