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
#calculate the spatial distribution of the average number of days of Baiu fronts in JJA (Fig.(d-f))
#======================================================

m1=6
m2=8
fn = '/Users/yiwenmao/Documents/StaFront2/biau/stcoord_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()

cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/bft/'
fn = cH+'HPB_ts.mat'
MC = sio.loadmat(fn)
dat6 = MC['dat6C']
dat7 = MC['dat7C']
dat8 = MC['dat8C']
#dat = dat6+dat7+dat8


fig,axsC=plt.subplots(1,3,figsize=(18,5),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
axsC=axsC.flatten()
proj=crs.PlateCarree()
axsC[0].coastlines(lw=0.5)
cmx = np.nanmean(dat6,axis=0)
cf = axsC[0].pcolor(lonn,latn,cmx,cmap='pink_r',transform=proj)
stdmx = np.nanstd(dat6,axis=0)
cs = axsC[0].contour(lonn,latn,stdmx,transform=proj)
axsC[0].clabel(cs,inline=True,fontsize=14)
cb=fig.colorbar(cf,orientation='vertical',pad=0.01,aspect=12,fraction=0.06,extend='max') 
axsC[0].set_title('mean annual days of Baiu Fronts (Jun)',fontsize=18)
cb.ax.tick_params(labelsize=14) 

axsC[1].coastlines(lw=0.5)
cmx = np.nanmean(dat7,axis=0)
stdmx = np.nanstd(dat7,axis=0)
cf = axsC[1].pcolor(lonn,latn,cmx,cmap='pink_r',transform=proj)
cs = axsC[1].contour(lonn,latn,stdmx,transform=proj)
axsC[1].clabel(cs,inline=True,fontsize=14)
cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='max') 
axsC[1].set_title('mean annual days of Baiu Fronts (Jul)',fontsize=18)
cb.ax.tick_params(labelsize=14) 

axsC[2].coastlines(lw=0.5)
cmx = np.nanmean(dat8,axis=0)
stdmx = np.nanstd(dat8,axis=0)
cf = axsC[2].pcolor(lonn,latn,cmx,cmap='pink_r',transform=proj)
cs = axsC[2].contour(lonn,latn,stdmx,transform=proj)
axsC[2].clabel(cs,inline=True,fontsize=14)
cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06,extend='max') 
axsC[2].set_title('mean annual days of Baiu Fronts (Aug)',fontsize=18)
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