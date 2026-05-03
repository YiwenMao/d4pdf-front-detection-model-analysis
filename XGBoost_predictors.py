import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import metpy.calc as mpcalc
import datetime
import math
import pickle
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import spearmanr
#================================================================================================
#This is a script for calculating the predictors of the XGBoost as described in Text S3 from ERA5
#================================================================================================

def getFtL(tmx):
    ck=tmx
    xs = np.where(ck)[0]
    ys = np.where(ck)[1]
    xs = np.unique(xs)
    ys = np.unique(ys)
    if len(xs)>1:
        dxsmax = np.diff(xs).max()
    else:
        dxsmax = 0

    if len(ys)>1:
        dysmax = np.diff(ys).max()
    else:
        dysmax = 0

    ftlx = len(xs)-dxsmax
    ftly = len(ys)-dysmax
    return ftlx,ftly,np.rad2deg(math.atan2(ftly,ftlx))

def harmonic_func(x,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10):
    params = [B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10]
    funcomp = [1,
               np.cos(ofreq*x),np.sin(ofreq*x),
               np.cos(2*ofreq*x),np.sin(2*ofreq*x),
               np.cos(3*ofreq*x),np.sin(3*ofreq*x), 
               np.cos(4*ofreq*x),np.sin(4*ofreq*x), 
               np.cos(5*ofreq*x),np.sin(5*ofreq*x),               
               ]
    return np.sum(np.array(params)*np.array(funcomp))

def harmonic_func1(x,B0,B1,B2,B3,B4,B5,B6):
    params = [B0,B1,B2,B3,B4,B5,B6]
    funcomp = [1,
               np.cos(ofreq*x),np.sin(ofreq*x),
               np.cos(2*ofreq*x),np.sin(2*ofreq*x),
               np.cos(3*ofreq*x),np.sin(3*ofreq*x)               
               ]
    return np.sum(np.array(params)*np.array(funcomp))

def harmonic_func2(x,B7,B8,B9,B10):
    params = [B7,B8,B9,B10]
    funcomp = [np.cos(4*ofreq*x),np.sin(4*ofreq*x), 
               np.cos(5*ofreq*x),np.sin(5*ofreq*x)            
               ]
    return np.sum(np.array(params)*np.array(funcomp))

def getbn():
    m1=6
    m2=8
    fn = 'stcoord_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    latn = MC['lat'].flatten()
    lonn = MC['lon'].flatten()

    fn = 'GridFront_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    Yft=MC['Yft']
    dtLC=MC['dtLC']
    Nt,nd1,nd2 = Yft.shape
    for i in np.arange(0,Nt):
        Yft[i,:,:]=np.flipud(Yft[i,:,:])

    isft = np.ones((Nt,),dtype=bool)
    for i in np.arange(0,Nt):
        if Yft[i,2:-2,2:-2].sum()>0:
            fmx = Yft[i,:,:]
            ftlx,ftly,tanft = getFtL(fmx)
            ll = np.max(np.array([ftlx,ftly]))
            if ll<=4:
                #if ll==4:
                #    print(i)
                isft[i]=False
        else:
            isft[i]=False

    fn = '/Users/yiwenmao/Documents/StaFront/JJA_kwd.csv'
    dfn = pd.read_csv(fn,index_col=0)
    bn = (dfn['baiu'].values)|(dfn['brain'].values)

    Nt = len(dfn)
    for i in np.arange(2,Nt-2):
        if (bn[i-2]==True)&(bn[i+2]==True):
            if isft[i]==True:
                dfn['baiu'].iloc[i]=True
            if isft[i-1]==True:
                dfn['baiu'].iloc[i-1]=True
            if isft[i+1]==True:
                dfn['baiu'].iloc[i+1]=True            

    bn = (dfn['baiu'].values)|(dfn['brain'].values)


    Nt,nd1,nd2 = Yft.shape
    Ydata = np.zeros((Nt,nd1,nd2,2),dtype=bool)
    Ydata[:,:,:,0] = True
    for i in np.arange(0,Nt):
        if isft[i]:
            Ydata[i,:,:,0]= Yft[i,:,:]==0 #no Front
            Ydata[i,:,:,1]= Yft[i,:,:]==1

    return bn,latn,lonn,dtLC,Yft,isft,Ydata
#=====================================================
bn,_,_,_,Yft,isft,Ydata = getbn()
MC = sio.loadmat('/Users/yiwenmao/Documents/StaFront2/biau/ncep2/stcoord.mat')
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()
vnm = 'vort' #choose the variable, either vorticity(vort) or MFC(dmoist)
#vnm='dmoist'
fn='/Users/yiwenmao/Documents/StaFront2/biau/ncep2/'+vnm+'_'+'850'+'.mat'
MC=sio.loadmat(fn)
data = MC['data']/1000
dtL = MC['dtL']
vdk = ~((dtL[:,1]==2)&(dtL[:,2]==29))
data = data[vdk,:,:]
dtL = dtL[vdk,:]

yrC = np.unique(dtL[:,0])
Nyr = len(yrC)
JJA_dayEY = np.arange(152,244)
dayY = np.arange(1,366)
Nssn = len(JJA_dayEY)

cmx = np.zeros((Nyr,Nssn))
for yi in np.arange(0,Nyr):
    cmx[yi,:] = JJA_dayEY
JJA_dayY = cmx.flatten()

cmx = np.zeros((Nyr,365))
for yi in np.arange(0,Nyr):
    cmx[yi,:] = dayY
dayCount = cmx.flatten()

ck1 = (latn>=30)&(latn<=40)
data = data[:,ck1,:]
#start from here
Nt,nd1,nd2 = data.shape
yts = np.reshape(data,(Nt,nd1*nd2))
yts = np.nanmean(yts,axis=1)

wL=11
warr = np.zeros((wL,))
warr[0] =0.027
warr[wL-1]=0.027
warr[1] = 0.05856
warr[wL-2]=0.05856
warr[2]=0.09030
warr[wL-3]=0.09030
warr[3]=0.11742
warr[wL-4]=0.11742
warr[4]=0.13567
warr[wL-5]=0.13567
warr[5]=0.142

Filter_boxcar = warr/wL
Filter_ts1 = signal.convolve(yts,Filter_boxcar)
Filter_ts1 = Filter_ts1[0:Nt]

pid=365
ofreq = 2*math.pi/pid

xs = np.arange(0,Nt)
popt, _ = curve_fit(harmonic_func, xs,Filter_ts1)
B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10 = popt

yts1 = np.zeros((Nt,)) #l-mode
yts2 = np.zeros((Nt,)) #s-mode
for i in np.arange(0,Nt):
    yts1[i]=harmonic_func1(xs[i],B0,B1,B2,B3,B4,B5,B6)-Filter_ts1.mean()
    #yts1[i]=harmonic_func1(xs[i],B0,B1,B2,B3,B4,B5,B6)
    yts2[i]=harmonic_func2(xs[i],B7,B8,B9,B10)

yts3 = yts1+yts2 #sum of l+s mode
yts4 = Filter_ts1 - Filter_ts1.mean() #from original reanalysis
#yts4 = Filter_ts1
yts5 = yts4 - yts1 #remove seasonal cycle (remove l mode)

yts1_dayY = np.zeros(dayY.shape)
yts2_dayY = np.zeros(dayY.shape)
yts3_dayY = np.zeros(dayY.shape)
yts4_dayY = np.zeros(dayY.shape)
yts5_dayY = np.zeros(dayY.shape)
ftC_dayY=np.zeros(dayY.shape)
for i in np.arange(0,len(dayY)):
    yts1_dayY[i]=np.nanmean(yts1[dayCount==dayY[i]])
    yts2_dayY[i]=np.nanmean(yts2[dayCount==dayY[i]])
    yts3_dayY[i]=np.nanmean(yts3[dayCount==dayY[i]])
    yts4_dayY[i]=np.nanmean(yts4[dayCount==dayY[i]])
    yts5_dayY[i] = np.nanmean(yts5[dayCount==dayY[i]])
    latft = np.flipud(latn)
    if (JJA_dayY==dayY[i]).sum()>0:
        Yft_day = Ydata[(JJA_dayY==dayY[i])&bn,:,:,1]
        #Yft_day = Yft_day[:,7:16,:]
        n1,n2,n3 = Yft_day.shape
        Yft_day = np.reshape(Yft_day,(n1,n2*n3))
        ftC_dayY[i]=(np.sum(Yft_day,axis=1)>0).sum()/21


d1=151
d2=243

JJA_yts1 = yts1_dayY[d1:d2] #l-mode sn fun
ymean = JJA_yts1.mean()
ymax = JJA_yts1.max()
ymin = JJA_yts1.min()
JJA_yts1 = (JJA_yts1 - ymean)/((ymax-ymin)/2)

JJA_yts2 = yts2_dayY[d1:d2] #s-mode sn fun
ymean = JJA_yts2.mean()
ymax = JJA_yts2.max()
ymin = JJA_yts2.min()
JJA_yts2 = (JJA_yts2 - ymean)/((ymax-ymin)/2)

JJA_yts3 = yts3_dayY[d1:d2] #l+s mode sn fun
ymean = JJA_yts3.mean()
ymax = JJA_yts3.max()
ymin = JJA_yts3.min()
JJA_yts3 = (JJA_yts3 - ymean)/((ymax-ymin)/2)

JJA_yts4 = yts4_dayY[d1:d2] #l+s (anomalies)
ymean = JJA_yts4.mean()
ymax = JJA_yts4.max()
ymin = JJA_yts4.min()
JJA_yts4 = (JJA_yts4 - ymean)/((ymax-ymin)/2)


JJA_yts5 = yts5_dayY[d1:d2] #s-mode (anomalies)
ymean = JJA_yts5.mean()
ymax = JJA_yts5.max()
ymin = JJA_yts5.min()
JJA_yts5 = (JJA_yts5 - ymean)/((ymax-ymin)/2)

JJA_ftC = ftC_dayY[d1:d2]
JJA_xs = np.arange(6,9,3/92)

# build date list
start = datetime.datetime(2026,6,1)
end = datetime.datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)

fig,axs = plt.subplots(1,1,figsize=(8,6))
axs.plot(dates,JJA_yts1,'b', label='l-mode')
axs.plot(dates,JJA_yts2,color='g',label = 's-mode')
axs.plot(dates,JJA_yts3,color='m', label = 'l-mode+s-mode')
axs.plot(dates,JJA_yts4,color='orange',label='averaged anomaly')
#axs.plot(dates,JJA_yts5,'m--', label='s-mode (Average)')
axs.set_xlabel('JJA dates', fontsize=18)
#axs.set_ylabel('Normalized (l+s or s) Functions',fontsize=16)
#axs.set_xlim([6,9])
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
rax=axs.twinx()
rax.plot(dates,JJA_ftC,'ro--',lw=2)
rax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
rax.set_ylabel('Biau Fronts Freq (day)',color='r',fontsize=16)
axs.legend(fontsize=16)
for lbl in rax.get_yticklabels():
    lbl.set_color('red')

#axs.set_title('normalized relative vorticity',fontsize=22)
axs.set_title('moisture flux convergence',fontsize=20)
plt.setp(axs.get_yticklabels(),fontsize=14)
plt.setp(axs.get_xticklabels(),fontsize=14)
plt.setp(rax.get_yticklabels(),fontsize=14)


print(spearmanr(JJA_ftC,JJA_yts5))
print(spearmanr(JJA_ftC,JJA_yts4))
print(spearmanr(JJA_yts5,JJA_yts4))

#save the predictors for futher use
outH='/Users/yiwenmao/Documents/StaFront2/biau/test_data/'
fout = outH+'ObsBiau_'+vnm+'_ref.mat'
sio.savemat(fout,{'JJA_xs':JJA_xs,'JJA_yts1':JJA_yts1,
                  'JJA_yts2':JJA_yts2,'JJA_yts3':JJA_yts3,
                  'JJA_yts4':JJA_yts4,'JJA_yts5':JJA_yts5,
                  'JJA_ft':JJA_ftC,'yts4':yts4,'yts5':yts5,'yts1':yts1,
                  'yts2':yts2,'yts3':yts3,
                  'dtL':dtL})
print(fout)
