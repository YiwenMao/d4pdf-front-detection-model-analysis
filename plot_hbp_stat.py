import numpy as np
import scipy.io as sio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import t as tdist
from scipy.stats import skew, kurtosis
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import skew, kurtosis

#======================================================
#calculate the spatial distribution of statistcs related to the average number of days of baiu/non-baiu fronts in JJA (Fig(g-l))
#======================================================
fn = 'data/ERA5_JJA_pred.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()

cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/bft/'
fn = cH+'HPB_ts.mat'
MC = sio.loadmat(fn)
dat6 = MC['dat6C']
dat7 = MC['dat7C']
dat8 = MC['dat8C']
dat = dat6+dat7+dat8

fig,axsC=plt.subplots(3,1,figsize=(5,12),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
axsC=axsC.flatten()
proj=crs.PlateCarree()
axsC[0].coastlines(lw=0.5)
cmx = np.nanmean(dat,axis=0)
cf = axsC[0].pcolor(lonn,latn,cmx,cmap='pink_r',transform=proj)
cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='max') 
axsC[0].set_title('mean annual days of non-Baiu Fronts',fontsize=14)

#axsC[1].coastlines(lw=0.5)
cmx = np.nanstd(dat,axis=0)
cs = axsC[0].contour(lonn,latn,cmx,transform=proj)
axsC[0].clabel(cs,inline=True,fontsize=12)
#cb=fig.colorbar(cf,orientation='vertical',pad=0.1,aspect=12,fraction=0.06,extend='max') 
#axsC[0].set_title('std annual days of non-Baiu Fronts',fontsize=14)
cb.ax.tick_params(labelsize=14) 

axsC[1].coastlines(lw=0.5)
cmx = skew(dat,axis=0)
cf = axsC[1].pcolor(lonn,latn,cmx,cmap='bwr',vmin=-2,vmax=2,transform=proj)
cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='max') 
axsC[1].set_title('skewness of annual days of non-Baiu Fronts',fontsize=14)
cb.ax.tick_params(labelsize=14) 

axsC[2].coastlines(lw=0.5)
cmx = kurtosis(dat,axis=0)
cf = axsC[2].pcolor(lonn,latn,cmx,cmap='rainbow',vmin=0,vmax=10,transform=proj)
cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='max') 
axsC[2].set_title('kurtosis of annual days of Baiu Fronts',fontsize=14)
cb.ax.tick_params(labelsize=14) 

for axs in axsC:
    axs.set_extent([lonn.min(),lonn.max(),latn.min(),latn.max()],crs=crs.PlateCarree())
    gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}