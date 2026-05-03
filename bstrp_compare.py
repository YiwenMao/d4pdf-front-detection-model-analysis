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
import math
#=========================================================================================
#bootstrap: resample all available years with replacement 10,000 times from the past climate (HPB) and from future climate (HFB 2K/4K with the following SST warming scenario (CC, GF, HA, MI, MP, MR) and calculate the difference in statistics between HPB and each HFB scenario for each bootstrap sample.
#=========================================================================================

#load the lat/lon info
m1=6
m2=8
fn = '/Users/yiwenmao/Documents/StaFront2/biau/stcoord_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()
#-------------------------------------------------------
vnm='bft'
exp_str='HPB'
fn ='result/'+exp_str+'_tmx.mat'
MC=sio.loadmat(fn)
dat_ref=MC['c'+vnm]
Nall,nd1,nd2 = dat_ref.shape
#cnbft=MC['cnbft']
indp = np.arange(0,Nall)
Nk=10000

tt = '2K'
dfstat=dict()
for sst in ['CC','GF','HA','MI','MP','MR']:
    cmx = np.zeros((Nk,nd1,nd2))
    dfstat[sst]=cmx

for k in np.arange(0,Nk):
    print(k)
    selected = np.random.choice(indp,size=Nall,replace=True)
    substat_ref = np.nanmean(dat_ref[selected,:,:],axis=0)
    for sst in ['CC','GF','HA','MI','MP','MR']:
        exp_str='HFB_'+tt+'_'+sst
        fn='result/'+exp_str+'_tmx.mat'
        MC = sio.loadmat(fn)
        dat=MC['c'+vnm]
        Nt,nd1,nd2 = dat.shape
        inda = np.arange(0,Nt)
        selected = np.random.choice(inda,size=len(inda),replace=True)
        substat = np.nanmean(dat[selected,:,:],axis=0)
        dfstat[sst][k,:,:]=substat_ref-substat
        #break
    #break

fout = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/HFB_'+tt+'_space_compare.mat'
sio.savemat(fout,dfstat)
print(fout)