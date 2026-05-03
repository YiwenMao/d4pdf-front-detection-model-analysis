import numpy as np
import scipy.io as sio
import pandas as pd
#import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier
import math
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import spearmanr
import metpy.calc as mpcalc
from tensorflow.keras.models import model_from_json
from scipy.ndimage import gaussian_filter
from datetime import datetime, timedelta
import sys
from pathlib import Path
#============================================================================================================
#This is a script to apply trained model 1 (U-NET) to d4PDF GCMs to get locations of fronts for each 
#============================================================================================================
def getNN(fd):
    json_file = open(fd+'.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(fd+'.weights.h5')
    return loaded_model

def getMoist(qu,qv,dy,dx):
    Nt,nd1,nd2 = qu.shape
    dqudx = np.zeros((Nt,nd1,nd2))
    dqvdy = np.zeros((Nt,nd1,nd2))
    for ti in np.arange(0,Nt):
        dqudx[ti,:,:],_ = mpcalc.gradient(qu[ti,:,:],deltas=(dy,dx))
        _,dqvdy[ti,:,:] = mpcalc.gradient(qv[ti,:,:],deltas=(dy,dx))
    dMoist = -1*(dqudx+dqvdy)
    return dMoist

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

#=======================================================
#load predictors calcualted from stored d4PDF dataset
#this is an example for HPB, the script for HFB 2K/4K experiments essentially have the same 
mdlH='/data12/yiwen/biauFront/'
datH925 = '/data12/yiwen/biauFront/d4PDF_mb925/HPB/'
datH850 = '/data12/yiwen/biauFront/d4PDF_mb850/HPB/'
datH700 = '/data12/yiwen/biauFront/d4PDF_mb700/HPB/'
fn = '/data12/yiwen/biauFront/d4PDFcd.mat'

MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()

dx, dy = mpcalc.lat_lon_grid_deltas(lonn, latn)
dx = dx/1000
dy = dy/1000

#mstr = 'm001'
mstr = sys.argv[1] #specify the ensemble member
#yr_str = '1952'
#yr_str = sys.argv[2]
#mon_str = '06'
#mon_str = sys.argv[3]

for cyr in np.arange(1952,2012): #loop over all years available for each ensemble

    for mon_str in ['06','07','08']:
        yr_str = str(int(cyr))
        #cyr = int(yr_str)
        cmon = int(mon_str)
        start = datetime(cyr,cmon,1)
        end = datetime(cyr, int(cmon+1), 1)-timedelta(hours=6)

        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(hours=6)

        Nt = len(dates)
        dtL = np.zeros((Nt,3))
        for i in np.arange(0,Nt):
            dtL[i,0] = dates[i].year
            dtL[i,1] = dates[i].month
            dtL[i,2] = dates[i].day

        fn = datH925+mstr+'_'+yr_str+mon_str+'.mat'
        file_path = Path(fn)
        if not file_path.exists():
            continue
        datC925 = sio.loadmat(fn)
        dataC=dict()
        dataC['q925'] = datC925['qsh']*1000
        dataC['u925'] = datC925['uwnd']
        dataC['v925'] = datC925['vwnd']
        dataC['w925'] = datC925['wwnd']
        dataC['vort925'] = datC925['vort']
        dataC['ept925'] = datC925['eptemp']

        fn = datH850+mstr+'_'+yr_str+mon_str+'.mat'
        file_path = Path(fn)
        if not file_path.exists():
            continue
        datC850 = sio.loadmat(fn)
        dataC['q850'] = datC850['qsh']*1000
        dataC['u850'] = datC850['uwnd']
        dataC['v850'] = datC850['vwnd']
        dataC['w850'] = datC850['wwnd']
        dataC['vort850'] = datC850['vort']
        dataC['ept850'] = datC850['eptemp']

        fn = datH700+mstr+'_'+yr_str+mon_str+'.mat'
        file_path = Path(fn)
        if not file_path.exists():
            continue
        datC700 = sio.loadmat(fn)
        dataC['q700'] = datC700['qsh']*1000
        dataC['u700'] = datC700['uwnd']
        dataC['v700'] = datC700['vwnd']
        dataC['w700'] = datC700['wwnd']
        dataC['vort700'] = datC700['vort']
        dataC['ept700'] = datC700['eptemp']

        Nt,d1n,d2n = datC700['uwnd'].shape
        datanC=dict()
        dField1C=dict()
        dField2C=dict()
        for vnm in ['w925','w850','w700','vort925','vort850','vort700','ept925','ept850','ept700']:
            datan = dataC[vnm]
            Field1C = np.zeros((Nt,d1n,d2n))
            Field2C = np.zeros((Nt,d1n,d2n))
            for ti in np.arange(0,Nt):
                tes = datan[ti,:,:]
                tes = gaussian_filter(tes,sigma=1)
                dtes_dy,dtes_dx = mpcalc.gradient(tes, deltas=(dy,dx))
                Field1=dtes_dx
                Field2=dtes_dy
                Field1C[ti,:,:]=Field1
                Field2C[ti,:,:]=Field2
            dField1C[vnm]=Field1C
            dField2C[vnm]=Field2C

        mask1CC=dict()
        eq3CC = dict()
        for vnm in ['ept925','ept850','ept700']:
            datan= dataC[vnm]
            Field1C = np.zeros((Nt,d1n,d2n))
            Field2C = np.zeros((Nt,d1n,d2n))
            Field3C = np.zeros((Nt,d1n,d2n))
            Field4C = np.zeros((Nt,d1n,d2n))
            Fieldmag12C = np.zeros((Nt,d1n,d2n))
            Fieldmag34C = np.zeros((Nt,d1n,d2n))
            mask1C = np.zeros((Nt,d1n,d2n))
            Field5C = np.zeros((Nt,d1n,d2n))
            Field6C = np.zeros((Nt,d1n,d2n))
            Field7C = np.zeros((Nt,d1n,d2n))
            Field8C = np.zeros((Nt,d1n,d2n))
            eq1C = np.zeros((Nt,d1n,d2n))
            eq3C = np.zeros((Nt,d1n,d2n))
            eq4C = np.zeros((Nt,d1n,d2n))     

            for ti in np.arange(0,Nt):
                tes = datan[ti,:,:] #starting from the surrogate temp variable
                tes = gaussian_filter(tes,sigma=1)

                dtes_dy,dtes_dx = mpcalc.gradient(tes, deltas=(dy,dx))
                Field1=dtes_dx
                Field2=dtes_dy
                dtes_mag = np.sqrt(dtes_dy**2+dtes_dx**2)
                Field_mag_12 = dtes_mag

                dtes_mag_dy,dtes_mag_dx = mpcalc.gradient(dtes_mag,deltas=(dy,dx))
                Field3 = dtes_mag_dx
                Field4 = dtes_mag_dy
                ddtest_mag_mag = np.sqrt(dtes_mag_dy**2+dtes_mag_dx**2)
                Field_mag_34 = ddtest_mag_mag

                vec1 = [Field3,Field4]
                vec2 = [Field1,Field2]/dtes_mag
                mask1 = -1*(vec1[0]*vec2[0]+vec1[1]*vec2[1])
                dF3_dy, dF3_dx = mpcalc.gradient(Field3,deltas=(dy,dx))
                dF4_dy, dF4_dy = mpcalc.gradient(Field4,deltas=(dy,dx))

                val=(-Field1*Field3-Field2*Field4)/np.sqrt(Field1**2+Field2**2)
                Field6,Field5 = mpcalc.gradient(val,deltas=(dy,dx)) 
                Field8,Field7 = mpcalc.gradient(np.sqrt(Field3**2+Field4**2),deltas=(dy,dx))

                eq1 = dF3_dx+dF4_dy
                eq4 = (Field1*Field5)/np.sqrt(Field1**2+Field2**2)+(Field6*Field2)/np.sqrt(Field1**2+Field2**2)
                eq3 = (Field7*Field3+Field8*Field4)/np.sqrt(Field3**2+Field4**2)

                Field1C[ti,:,:]=Field1
                Field2C[ti,:,:]=Field2
                Field3C[ti,:,:]=Field3
                Field4C[ti,:,:]=Field4
                Field5C[ti,:,:]=Field5
                Field6C[ti,:,:]=Field6
                Field7C[ti,:,:]=Field7
                Field8C[ti,:,:]=Field8
                Fieldmag12C[ti,:,:]=Field_mag_12
                Fieldmag34C[ti,:,:]=Field_mag_34
                mask1C[ti,:,:]=mask1

                eq1C[ti,:,:]=eq1
                eq3C[ti,:,:]=eq3
                eq4C[ti,:,:]=eq4

            mask1CC[vnm] = mask1C
            eq3CC[vnm] = eq3C

        qu925 = dataC['q925']*dataC['u925']
        qv925 = dataC['q925']*dataC['v925']
        qu850 = dataC['q850']*dataC['u850']
        qv850 = dataC['q850']*dataC['v850']
        qu700 = dataC['q700']*dataC['u700']
        qv700 = dataC['q700']*dataC['v700']

        Xdata = np.zeros((Nt,d1n,d2n,18))
        Xdata[:,:,:,0]=getMoist(qu925,qv925,dy,dx)
        Xdata[:,:,:,1]=getMoist(qu850,qv850,dy,dx)
        Xdata[:,:,:,2]=getMoist(qu700,qv700,dy,dx)
        Xdata[:,:,:,3]=dField2C['w700']
        Xdata[:,:,:,4]=dField2C['w850']
        Xdata[:,:,:,5]=dField2C['w925']
        Xdata[:,:,:,6]=dField2C['vort700']
        Xdata[:,:,:,7]=dField2C['vort850']
        Xdata[:,:,:,8]=dField2C['vort925']
        Xdata[:,:,:,9]=dField2C['ept700']
        Xdata[:,:,:,10]=dField2C['ept850']
        Xdata[:,:,:,11]=dField2C['ept925']
        Xdata[:,:,:,12]=eq3CC['ept925']
        Xdata[:,:,:,13]=mask1CC['ept925']
        Xdata[:,:,:,14]=eq3CC['ept850']
        Xdata[:,:,:,15]=mask1CC['ept850']
        Xdata[:,:,:,16]=eq3CC['ept700']
        Xdata[:,:,:,17]=mask1CC['ept700']

        _,_,_,Np = Xdata.shape
        for k in np.arange(0,Np):
            vmin = np.nanmin(Xdata[:,:,:,k])
            vmax = np.nanmax(Xdata[:,:,:,k])
            Xdata[:,:,:,k] = 2*(Xdata[:,:,:,k]-vmin)/(vmax-vmin)-1   

        #--------------------------------------------------------------------
        mdl_type='ftNft'
        mdlyrC = np.arange(2000,2021)
        Nmdl = len(mdlyrC)

        ftPredC1 = []
        ftPredC3 = []
        for testyr in mdlyrC: #load all trained models, the final prediction is based on vote by majority
            vnm = 'new1'
            fn = mdlH+'unet/'+mdl_type+'/'+vnm+'/build1_'+str(testyr)
            mdl = getNN(fn)
            ftyr = mdl.predict(Xdata[:,:,:,0:12])
            ftyr = ftyr[:,:,:,1]>0.5
            ftPredC1.append(ftyr)

            vnm = 'new3'
            fn = mdlH+'unet/'+mdl_type+'/'+vnm+'/build1_'+str(testyr)
            mdl = getNN(fn)
            ftyr = mdl.predict(Xdata)
            ftyr = ftyr[:,:,:,1]>0.5
            ftPredC3.append(ftyr)

        Nt,_,_=ftyr.shape
        thres = int((Nmdl/2)+0.5)
        dout1 = np.zeros((Nt,d1n,d2n),dtype=bool)
        for i in np.arange(0,Nt):
            cft = np.zeros((Nmdl,d1n,d2n))
            for m in np.arange(0,Nmdl):
                cft[m,:,:]=ftPredC1[m][i,:,:]
            ftmx = np.sum(cft,axis=0)>thres
            dout1[i,:,:]=ftmx

        dout3 = np.zeros((Nt,d1n,d2n),dtype=bool)
        for i in np.arange(0,Nt):
            cft = np.zeros((Nmdl,d1n,d2n))
            for m in np.arange(0,Nmdl):
                cft[m,:,:]=ftPredC3[m][i,:,:]
            ftmx = np.sum(cft,axis=0)>thres
            dout3[i,:,:]=ftmx

                #break
            #break

        FtPredOut_org = (dout1)|(dout3)
        isftNL,FtPredOut_isft = getIsFt(FtPredOut_org)

        #here I store the prediction results with models with different hyperparamters
        #but dout3 is the described in the main text, which is the one I used for further data analysis 
        fout = mdlH+'d4PDF_Ft/HPB/'+mstr+'_'+yr_str+'_'+mon_str+'.mat'
        MC = sio.savemat(fout,{'isft':isftNL,
                            'FtOut_org':FtPredOut_org,
                            'FtOut_isft':FtPredOut_isft,
                            'data1':dout1,
                            'data3':dout3,
                            'time':dtL})
        print(fout)

        #break