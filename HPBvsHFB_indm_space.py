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

#---------------------------------------------------------
#space difference (HFB-HPB) for individual models for JJA, a script to get Fig9
#---------------------------------------------------------
m1=6
m2=8
fn = '/Users/yiwenmao/Documents/StaFront2/biau/stcoord_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()

tt = '4K'
vnm='bft'
datHFB=np.zeros((400,24,24))
datHFB_stdI = dict()
for sst in ['CC','GF','HA','MI','MP','MR']:
    fn = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/HFB_'+tt+'_'+sst+'_ts.mat'
    MC = sio.loadmat(fn)
    cdat=(MC['dat6C']+MC['dat7C']+MC['dat8C'])/3
    datHFB=datHFB+cdat
    datHFB_stdI[sst]=np.nanstd(cdat,axis=0)
    #break
datHFB = datHFB/6
datHFB_std = np.nanstd(datHFB,axis=0)

fn = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/HFB_'+tt+'_space_compare.mat'
MC = sio.loadmat(fn)

CImean=dict()
CIup=dict()
CIdn=dict()
CItest = dict()


CIm = np.zeros((24,24))
CIt = np.zeros((24,24))
for sst in ['CC','GF','HA','MI','MP','MR']:
    cdat = MC[sst]
    mean_cmx = np.nanmean(cdat,axis=0)
    up_cmx = np.percentile(cdat,97.5,axis=0)
    dn_cmx = np.percentile(cdat,2.5,axis=0)
    CImean[sst] = mean_cmx
    CIup[sst] = up_cmx
    CIdn[sst] = dn_cmx
    CItest[sst] = ~((up_cmx>0)&(dn_cmx<0))

    CIm = CIm+mean_cmx
    CIt = CIt+(~((up_cmx>=0)&(dn_cmx<=0)))
    #break

CIm = CIm/6
CIsig = (CIt>=4)
#===================================================================================

proj = crs.PlateCarree()
fig,axsC = plt.subplots(3,2,figsize=(12,15),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
axsC = axsC.flatten()
for knm,axs in zip(['CC','GF','HA','MI','MP','MR'],axsC):
    cmx = CImean[knm]
    cf = axs.pcolor(lonn,latn,-1*cmx,vmin=-3,vmax=3,cmap='bwr',transform=proj)

    cv = CItest[knm]
    rowInd = np.where(cv)[0]
    colInd = np.where(cv)[1]
    for k in np.arange(0,len(rowInd)):
        axs.plot(lonn[colInd[k]],latn[rowInd[k]],'k+',transform=proj)    

    cs = axs.contour(lonn,latn,datHFB_stdI[knm],levels=5,colors='k',transform=proj)
    axs.clabel(cs,inline=True,fontsize=13)
    axs.set_title('HFB '+tt+' '+knm+'-HPB',fontsize=18)
    axs.coastlines(lw=0.5)
    gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)
    axs.set_extent([lonn.min(),lonn.max(),latn.min(),latn.max()],crs=crs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='both')
    cb.ax.tick_params(labelsize=14)   
