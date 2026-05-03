import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import xgboost as xgb
import pandas as pd
import cartopy.crs as crs
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
#===========================================================
#a script to calculate F1, Cohen's Kappy and MCC to evalaute the U-Net models, produce Fig.2(a-f)
#===========================================================
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

def getMet(cc,ytest,ypred):
    OT=ytest==cc
    PT=ypred==cc
    OF=ytest!=cc
    PF=ypred!=cc
    A=(OT&PT).sum()
    B=(PT&OF).sum()
    C=(PF&OT).sum()
    D=(PF&OF).sum()

    ACC = (A+D)/(A+B+C+D)
    POD = A/(A+C)
    POFD = B/(B+D)
    FAR = B/(A+B)
    SR = A/(A+B)
    CSI = A/(A+B+C)
    bias = (A+B)/(A+C)

    return ACC,POD,POFD,FAR,SR,CSI,bias

def getIsFt(ftmxC):

    Nt,_,_ = ftmxC.shape
    isft = np.ones((Nt,),dtype=bool)
    for i in np.arange(0,Nt):
        fmx = ftmxC[i,:,:]
        if ftmxC[i,2:-2,2:-2].sum()>0:
            fmx = ftmxC[i,:,:]
            ftlx,ftly,tanft = getFtL(fmx)
            ll = np.max(np.array([ftlx,ftly]))
            if ll<=4:
                #if ll==4:
                #    print(i)
                isft[i]=False
        else:
            isft[i]=False

    Nt,nd1,nd2 = ftmxC.shape
    Ydata = np.zeros((Nt,nd1,nd2,2),dtype=bool)
    Ydata[:,:,:,0] = True
    for i in np.arange(0,Nt):
        if isft[i]:
            Ydata[i,:,:,0]= ftmxC[i,:,:]==0 #no Front
            Ydata[i,:,:,1]= ftmxC[i,:,:]==1

    return isft,Ydata[:,:,:,1]


def getbn():
    m1=6
    m2=8
    dH = '/Users/yiwenmao/Documents/StaFront2/biau/'
    fn = dH+'stcoord_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    latn = MC['lat'].flatten()
    lonn = MC['lon'].flatten()

    fn = dH+'GridFront_'+str(m1)+'_'+str(m2)+'.mat'
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

def getSMet(Ydata,YPred): #metrics before adjustment
    nl,nd1,nd2 = Ydata.shape
    ACC_a = np.zeros((nd1,nd2,2))
    POD_a = np.zeros((nd1,nd2,2))
    SR_a = np.zeros((nd1,nd2,2))
    CSI_a = np.zeros((nd1,nd2,2))
    bias_a = np.zeros((nd1,nd2,2))
    for i in np.arange(0,nd1):
        for j in np.arange(0,nd2):
            yo = Ydata[:,i,j]
            yp = YPred[:,i,j]
            ACC_a[i,j,0],POD_a[i,j,0],POFD,FAR,SR_a[i,j,0],CSI_a[i,j,0],bias_a[i,j,0]=getMet(True,yo,yp)
            ACC_a[i,j,1],POD_a[i,j,1],POFD,FAR,SR_a[i,j,1],CSI_a[i,j,1],bias_a[i,j,1]=getMet(False,yo,yp)                        
            #break
        #break 
    
    f1mx = np.zeros((24,24))
    f1mx[:] = np.nan
    kappa_mx = np.zeros((24,24))
    kappa_mx[:] = np.nan
    mcc_mx = np.zeros((24,24))
    mcc_mx[:] = np.nan
    for i in np.arange(0,24):
        for j in np.arange(0,24):
            yo = Ydata[:,i,j]
            yp = YPred[:,i,j]
            f1mx[i,j] = f1_score(yo, yp)
            kappa_mx[i,j] = cohen_kappa_score(yo, yp)
            mcc_mx[i,j] = matthews_corrcoef(yo, yp)

    return ACC_a,POD_a,SR_a,CSI_a,bias_a,f1mx,kappa_mx,mcc_mx

def getSMet1(Ydata,YPred): #metrics after adjustment
    nl,nd1,nd2 = Ydata.shape
    for ti in np.arange(0,nl):
        for j in np.arange(1,nd1-1):
            for k in np.arange(1,nd2-1):
                if Ydata[ti,j,k]==True:
                    ck0=YPred[ti,j,k]
                    ck1=YPred[ti,j+1,k+1]
                    ck2=YPred[ti,j+1,k]
                    ck3=YPred[ti,j+1,k+1]
                    ck4=YPred[ti,j,k-1]
                    ck5=YPred[ti,j,k+1]
                    ck6=YPred[ti,j-1,k-1]
                    ck7=YPred[ti,j-1,k]
                    ck8=YPred[ti,j-1,k+1]
                    YPred[ti,j,k]=ck0|ck1|ck2|ck3|ck4|ck5|ck6|ck7|ck8

    ACC_a = np.zeros((nd1,nd2,2))
    POD_a = np.zeros((nd1,nd2,2))
    SR_a = np.zeros((nd1,nd2,2))
    CSI_a = np.zeros((nd1,nd2,2))
    bias_a = np.zeros((nd1,nd2,2))
    for i in np.arange(0,nd1):
        for j in np.arange(0,nd2):
            yo = Ydata[:,i,j]
            yp = YPred[:,i,j]
            ACC_a[i,j,0],POD_a[i,j,0],POFD,FAR,SR_a[i,j,0],CSI_a[i,j,0],bias_a[i,j,0]=getMet(True,yo,yp)
            ACC_a[i,j,1],POD_a[i,j,1],POFD,FAR,SR_a[i,j,1],CSI_a[i,j,1],bias_a[i,j,1]=getMet(False,yo,yp)                        
            #break
        #break 

    f1mx = np.zeros((24,24))
    f1mx[:] = np.nan
    kappa_mx = np.zeros((24,24))
    kappa_mx[:] = np.nan
    mcc_mx = np.zeros((24,24))
    mcc_mx[:] = np.nan
    for i in np.arange(0,24):
        for j in np.arange(0,24):
            yo = Ydata[:,i,j]
            yp = YPred[:,i,j]
            f1mx[i,j] = f1_score(yo, yp)
            kappa_mx[i,j] = cohen_kappa_score(yo, yp)
            mcc_mx[i,j] = matthews_corrcoef(yo, yp)
    
    return ACC_a,POD_a,SR_a,CSI_a,bias_a,f1mx,kappa_mx,mcc_mx

#===================================================

fn = 'data/ERA5_JJA_pred.mat'
MC = sio.loadmat(fn)
FtObs=MC['FtObs']
FtPred=MC['FtPred']
lonn=MC['lon'].flatten()
latn=MC['lat'].flatten()

#ACC_a,POD_a,SR_a,CSI_a,bias_a = getSMet(Ydata,FtPred)
ACC_a,POD_a,SR_a,CSI_a,bias_a,f1mx,kappa_mx,mcc_mx = getSMet(FtObs,FtPred)
_,_,_,_,_,f1mx1,kappa_mx1,mcc_mx1 = getSMet1(FtObs,FtPred)
_,nd1,nd2 = FtObs.shape
freqFtO_a = np.sum(FtObs,axis=0)/len(FtObs)
freqFtP_a = np.sum(FtPred,axis=0)/len(FtObs)


proj = crs.PlateCarree()
fig,axsC=plt.subplots(3,2,figsize=(12,16),subplot_kw={'projection': crs.PlateCarree(central_longitude=180)})
axsC=axsC.flatten()
#cmx = SRsC[yr][:,:,0]
#cmx = CSI_a[:,:,0]
#cmx = bias_a[:,:,0]
for axs,cmx,knm in zip(axsC,[f1mx,f1mx1,kappa_mx,kappa_mx1,mcc_mx,mcc_mx1],
                       ['F1','Adjusted F1','Kappa','Adjusted Kappa','MCC','Adjusted MCC']):
    #cf=axs.contourf(lonn,latn,cmx,levels=np.arange(0,1.1,0.1),cmap='jet',transform=proj)
    cf=axs.pcolor(lonn,latn,cmx,vmin=0,vmax=1,cmap='rainbow',transform=proj)
    if knm =='Kappa':
        axs.set_title('Cohen\'s Kappa',fontsize=22)
    else:
        axs.set_title(knm,fontsize=22)

    axs.coastlines(lw=0.5)
    gl = axs.gridlines(crs=crs.PlateCarree(),draw_labels=True)
    axs.set_extent([lonn.min(),lonn.max(),latn.min(),latn.max()],crs=crs.PlateCarree())
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    cb=fig.colorbar(cf,orientation='vertical',pad=0.05,aspect=12,fraction=0.06)
    cb.ax.tick_params(labelsize=14)