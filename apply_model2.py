import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
import pandas as pd
#import xgboost as xgb
from datetime import datetime
#from sklearn.ensemble import RandomForestClassifier
import math
import pickle
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import spearmanr
import metpy.calc as mpcalc
#from tensorflow.keras.models import model_from_json
#import random
#import scipy.stats as sstat
#from sklearn.metrics import roc_auc_score, roc_curve
#from datetime import datetime, timedelta
#import matplotlib.dates as mdates
from pathlib import Path

#===========================================================================================================================
#This is a script for applying the trained XGBoost model to idenify days of Baiu Fronts and collecte the spatial locations of
# all Baiu Fronts
#===========================================================================================================================

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

def getMoist(qu,qv,dy,dx):
    Nt,nd1,nd2 = qu.shape
    dqudx = np.zeros((Nt,nd1,nd2))
    dqvdy = np.zeros((Nt,nd1,nd2))
    for ti in np.arange(0,Nt):
        dqudx[ti,:,:],_ = mpcalc.gradient(qu[ti,:,:],deltas=(dy,dx))
        _,dqvdy[ti,:,:] = mpcalc.gradient(qv[ti,:,:],deltas=(dy,dx))
    dMoist = dqudx+dqvdy
    return -1*dMoist
#=====================================================
#dH='/Users/yiwenmao/Documents/StaFront2/biau/var_model/'
#dH = '/external12/yiwen/StaFront2/biauFront/var_model/'
dH = '/external14/yiwen/biauFront/var_model/'
#dH = '/data12/yiwen/biauFront/var_model/'

#datH='/Users/yiwenmao/Documents/StaFront2/biau/'
datH='/external14/yiwen/biauFront/'
#datH = '/data12/yiwen/biauFront/'
#datH = '/external12/yiwen/StaFront2/biauFront/'

fn = datH+'stcoord_'+str(6)+'_'+str(8)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()
dx, dy = mpcalc.lat_lon_grid_deltas(lonn, latn)
dx = dx/1000
dy = dy/1000

#mstrC=['m101','m102','m103','m104','m105','m106','m107','m108','m109']

#mstrC = ['m001','m002','m003','m004','m005','m006',
#         'm007','m008','m009','m010','m011','m012']

#mstrC = ['m021','m022','m023','m024','m025','m026',
#         'm027','m028','m029','m030',
#         'm041','m042','m043','m044','m045','m046',
#         'm047','m048','m049','m050']

#ensemble members
mstrC  = ['m061','m062','m063','m064','m065','m066',
         'm067','m068','m069','m070',
         'm081','m082','m083','m084','m085','m086',
         'm087','m088','m089','m090']

#experiment
expstr = 'HPB'
yrbeg = 1952
yrend = 2012
#estr1,estr2, estr3 = expstr.split('_')
#if estr2 == '2K':
#    yrbeg = 2031
#    yrend = 2092

#if estr2 == '4K':
#    yrbeg = 2052
#    yrend = 2112

#p1
#fnmC = ['2001','2002','2003',
#        '2005','2006','2007','2008','2009','2010','2012','2014','2015',
#        '2016','2017','2018','2020','all']

#p2
fnmC = ['2001','2002','2003',
        '2005','2006','2007','2008','2009','2010','2011',
        '2012','2013','2014','2015',
        '2016','2017','2020'] #trained models marked by their year of training

#load all models 
clfC = []
for fn in fnmC:
    mdlf = dH+'xgboost_isbiau_from_bnpredMdl_p2_'+fn+'.pkl'
    with open(mdlf,'rb') as f:
        clf = pickle.load(f)
    clfC.append(clf)
Nclf = len(clfC)

dJJA_yts3 = dict()
dJJA_yts4 = dict()
dyts4 = dict()
for vnm in ['vort','dmoist']:
    fn = 'test_data/HPB/'+'HarmonicData_'+vnm+'.mat'
    MC = sio.loadmat(fn)
    dJJA_yts3[vnm] = MC['dJJA_yts3']
    dJJA_yts4[vnm] = MC['dJJA_yts4']

    fn = 'test_data/HPB/'+'HarmonicData_'+vnm+'_dyts4.csv'
    dfn = pd.read_csv(fn,index_col=0)
    Nt = len(dfn)
    dtL = np.zeros((Nt,3))
    for i in np.arange(0,Nt):
        tstr = dfn.index[i].split('-')
        yr = int(tstr[0])
        mon = int(tstr[1])
        day = int(tstr[2])
        dtL[i,0] = yr
        dtL[i,1] = mon
        dtL[i,2] = day
    
    tk = (dtL[:,1] == 6)|(dtL[:,1]==7)|(dtL[:,1]==8)
    dfn = dfn[tk]
    dyts4[vnm] = dfn
    dtL = dtL[tk,:]
#-------------------------------------------------------
JJA_xs = np.arange(6,9,3/92)
dayY = np.arange(1,366)
JJA_dayY = dayY[151:243]

Nssn = len(JJA_xs)
yrC = np.unique(dtL[:,0])
Nyr = len(yrC)
Nt,Nm = dfn.shape

cmx = np.zeros((Nyr,Nssn))
for yi in np.arange(0,Nyr):
    cmx[yi,:]=JJA_dayY
dayCount = cmx.flatten()

dtList = []
for ti in np.arange(0,len(dtL)):
    cdt = datetime(int(dtL[ti,0]),int(dtL[ti,1]), int(dtL[ti,2]))
    dtList.append(cdt)

for i in np.arange(0,len(mstrC)):
    #for each ensemble member
    ptor_JJA_yts3 = dict()
    ptor_JJA_yts4 = dict()
    ptor_yts4 = dict()
    vkC=[]
    for vnm in ['vort','dmoist']:

        ptor_yts4[vnm] = dyts4[vnm].iloc[:,i].values
        vkk = ~np.isnan(ptor_yts4[vnm])
        vkC.append(vkk)
        ys = ptor_yts4[vnm]
        ymean = np.nanmean(ys)
        ymax = np.nanmax(ys)
        ymin = np.nanmin(ys)
        ptor_yts4[vnm] = (ptor_yts4[vnm] - ymean)/((ymax-ymin)/2)

        cmx = np.zeros((Nyr,Nssn))
        for yi in np.arange(0,Nyr): 
            cmx[yi,:] = dJJA_yts3[vnm][:,i]
        ptor_JJA_yts3[vnm]=cmx.flatten()

        cmx = np.zeros((Nyr,Nssn))
        for yi in np.arange(0,Nyr):
            cmx[yi,:] = dJJA_yts4[vnm][:,i]
        ptor_JJA_yts4[vnm]=cmx.flatten()
        #break

    vk = vkC[0]&vkC[1]
    XdataFt_all = np.zeros((len(vk),6))
    XdataFt_all[:,0]=ptor_JJA_yts3['vort']
    XdataFt_all[:,1]=ptor_JJA_yts3['dmoist']
    XdataFt_all[:,2]=ptor_JJA_yts4['vort']
    XdataFt_all[:,3]=ptor_JJA_yts4['dmoist']
    XdataFt_all[:,4]=ptor_yts4['vort']
    XdataFt_all[:,5]=ptor_yts4['dmoist']
    #XdataFt_all = XdataFt_all[vk,:]
    mb = mstrC[i]


    YdataM=[] #predicted Baiu Fronts
    dtLOut=[] #date of Baiu Fronts
    for yr in np.arange(yrbeg,yrend):
        ttk = dtL[:,0]==yr
        if (ttk&vk).sum()==0:
            continue
        XdataFt = XdataFt_all[ttk&vk,:]
        cdtL = dtL[ttk&vk,:]

        #q850=[]
        #ept850=[]
        q850=[]
        u850=[]
        v850=[]
        vort850=[]
        for mon in ['06','07','08']:

            fn = datH+'d4PDF_mb850/'+expstr+'/'+mb+'_'+str(yr)+mon+'.mat'
            file_path = Path(fn)
            if not file_path.exists():
                continue

            MC = sio.loadmat(fn)
            u850.append(MC['uwnd'])
            v850.append(MC['vwnd'])
            q850.append(MC['qsh'])
            vort850.append(MC['vort'])

        #q700=[]
        #ept700=[]
        fdtL = []
        q700=[]
        u700=[]
        v700=[]
        vort700=[]
        for mon in ['06','07','08']:
            fn = datH+'d4PDF_mb700/'+expstr+'/'+mb+'_'+str(yr)+mon+'.mat'
            file_path = Path(fn)
            if not file_path.exists():
                continue
            MC = sio.loadmat(fn)
            q700.append(MC['qsh'])
            u700.append(MC['uwnd'])
            v700.append(MC['vwnd'])
            vort700.append(MC['vort'])
            
        q850 = np.concatenate(q850[:],axis=0)
        q700 = np.concatenate(q700[:],axis=0)
        u850 = np.concatenate(u850[:],axis=0)
        u700 = np.concatenate(v700[:],axis=0)
        v850 = np.concatenate(v850[:],axis=0)
        v700 = np.concatenate(v700[:],axis=0)
        vort850 = np.concatenate(vort850[:],axis=0)
        vort700 = np.concatenate(vort700[:],axis=0)

        qu850 = 1000*q850*u850
        qv850 = 1000*q850*v850
        qu700 = 1000*q700*u700
        qv700 = 1000*q700*v700
        dmoist850 = getMoist(qu850,qv850,dy,dx)
        dmoist700 = getMoist(qu700,qv700,dy,dx)

        Nt,nd1,nd2 = q850.shape

        predNLc=[]
        isftNL=[]
        fdtL = []
        for mon in ['06','07','08']:
            fn = datH+ 'd4PDF_Ft/'+expstr+'/'+mb+'_'+str(yr)+'_'+mon+'.mat'
            file_path = Path(fn)
            if not file_path.exists():
                continue
            MC = sio.loadmat(fn)
            predNLc.append(MC['FtOut_isft'])
            isftNL.append(MC['isft'].flatten()==1)
            fdtL.append(MC['time'])

            #break
        predNLc = np.concatenate(predNLc[:],axis=0)
        isftNL = np.concatenate(isftNL[:],axis=0)
        fdtL = np.concatenate(fdtL[:],axis=0)

        Nt1 = len(fdtL)
        Nt2 = len(cdtL)
        eFtL1 = np.zeros((Nt1,1))
        eFtAM1 = np.zeros((Nt1,4))
        for j in np.arange(0,Nt1):
            cmx = predNLc[j,:,:]==1
            if isftNL[j]:
                _,eFtL1[j],_=getFtL(cmx)
                eFtAM1[j,0] = dmoist850[j,:,:][cmx].mean()
                eFtAM1[j,1] = vort850[j,:,:][cmx].mean()
                eFtAM1[j,2] = dmoist700[j,:,:][cmx].mean()
                eFtAM1[j,3] = vort700[j,:,:][cmx].mean()                
            #break

        eFtL2 = np.zeros((Nt2,1))
        eFtL2[:]=np.nan
        eFtAM2 = np.zeros((Nt2,4))
        eFtAM2[:]=np.nan
        iseftam2 = np.zeros((Nt2,),dtype=bool)
        for j in np.arange(0,Nt2):
            dck = (fdtL[:,0]==cdtL[j,0])&(fdtL[:,1]==cdtL[j,1])&(fdtL[:,2]==cdtL[j,2])
            if dck.sum()>0:
                eFtAM2[j,:]=np.nanmean(eFtAM1[dck,:],axis=0) #physical properties of the predicted fronts (as potential predictors, but later dropped)
                eFtL2[j,:]=np.nanmax(eFtL1[dck,:],axis=0) #length of the predicted fronts
                iseftam2[j]=True
            #break

        Xdata = np.concatenate((XdataFt,eFtL2),axis=1) #the predictors
        Xdata = Xdata[iseftam2,:]
        nx1,nx2 = Xdata.shape
        Ydata = np.zeros((nx1,2),dtype=bool)
        YdataC = np.zeros((nx1,Nclf))
        for ci in np.arange(0,Nclf):
            eYdata = clf.predict(Xdata) #prediction by individual models
            YdataC[:,ci]=eYdata[:,1]
        thd = int((Nclf/2)+0.5)
        Ydata = np.sum(YdataC,axis=1)>thd #the final prediction is based on vote by majority
        YdataM.append(Ydata)
        dtLOut.append(cdtL[iseftam2,:])   
        #break

    YdataM = np.concatenate(YdataM[:],axis=0)
    dtLOut = np.concatenate(dtLOut[:],axis=0)
    fout = 'd4PDF_bFt_p2/'+expstr+'/'+mb+'.mat'
    sio.savemat(fout,{'dtL':dtLOut,'data':YdataM})
    print(fout)
    #break