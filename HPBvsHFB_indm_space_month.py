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
#--------------------------------------------------------------------------------------
#calculate the space difference (HFB-HPB) for individual models for each month, a script to get Fig(6-8)
#--------------------------------------------------------------------------------------
m1=6
m2=8
fn = '/Users/yiwenmao/Documents/StaFront2/biau/stcoord_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()
vnm='bft'
ms='8' #choose the month
mnm = 'dat'+ms+'C'
exp_str='HPB'
fn ='/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/'+exp_str+'_ts.mat'
MC = sio.loadmat(fn)
dat_ref=MC[mnm]

tt = '4K'
datHFB=np.zeros((400,24,24))
datHFB_stdI = dict()
for sst in ['CC','GF','HA','MI','MP','MR']:
    fn = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/HFB_'+tt+'_'+sst+'_ts.mat'
    MC = sio.loadmat(fn)
    datHFB=datHFB+MC[mnm]
    datHFB_stdI[sst]=np.nanstd(MC[mnm],axis=0)
    #break
datHFB = datHFB/6
datHFB_std = np.nanstd(datHFB,axis=0)

fn = '/Users/yiwenmao/Documents/StaFront3/result/YearData/'+vnm+'/HFB_'+tt+'_'+ms+'_ts_compare.mat'
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
    CItest[sst] = ~((up_cmx>=0)&(dn_cmx<=0))
    CIm = CIm+mean_cmx
    CIt = CIt+(~((up_cmx>=0)&(dn_cmx<=0)))
    #break

'''
CIm = CIm/6
CIsig = (CIt>=4)
proj = crs.PlateCarree()
fig,axs = plt.subplots(1,1,figsize=(8,6),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
cmx = CIm
cf = axs.pcolor(lonn,latn,-1*cmx,vmin=-0.2,vmax=0.2,cmap='bwr',transform=proj)
axs.coastlines(lw=0.5)
gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)

cv = CIt>=4
rowInd = np.where(cv)[0]
colInd = np.where(cv)[1]
for k in np.arange(0,len(rowInd)):
    axs.plot(lonn[colInd[k]],latn[rowInd[k]],'k+',alpha=0.5,transform=proj)   

cv = CIt==6
rowInd = np.where(cv)[0]
colInd = np.where(cv)[1]
for k in np.arange(0,len(rowInd)):
    axs.plot(lonn[colInd[k]],latn[rowInd[k]],'ko',mfc='none',alpha=0.5,transform=proj)
cs = axs.contour(lonn,latn,datHFB_std,levels=5,transform=proj)
axs.clabel(cs,inline=True,fontsize=10)

axs.set_title('HFB '+tt+' (MultiM)-HPB',fontsize=14)
axs.coastlines(lw=0.5)
gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)
axs.set_extent([lonn.min(),lonn.max(),latn.min(),latn.max()],crs=crs.PlateCarree())
gl.top_labels = False
gl.right_labels = False
gl.left_labels = True
gl.bottom_labels = True
cb=fig.colorbar(cf,orientation='vertical',pad=0.1,aspect=12,fraction=0.06,extend='both')
'''
#===================================================================================

proj = crs.PlateCarree()
fig,axsC = plt.subplots(3,2,figsize=(12,15),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
axsC = axsC.flatten()
for knm,axs in zip(['CC','GF','HA','MI','MP','MR'],axsC):
    cmx = CImean[knm]
    cf = axs.pcolor(lonn,latn,-1*cmx,vmin=-0.2,vmax=0.2,cmap='bwr',transform=proj)
    #axs.coastlines(lw=0.5)
    #gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)

    cv = CItest[knm]
    rowInd = np.where(cv)[0]
    colInd = np.where(cv)[1]
    for k in np.arange(0,len(rowInd)):
        axs.plot(lonn[colInd[k]],latn[rowInd[k]],'k+',transform=proj,alpha=0.5)    

    cs = axs.contour(lonn,latn,datHFB_stdI[knm],levels=5,colors = 'k',transform=proj)
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
    #axs.tick_params(axis='both', labelsize=16)   