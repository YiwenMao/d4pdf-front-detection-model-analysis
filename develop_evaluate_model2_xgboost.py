import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
#from sklearn.ensemble import RandomForestClassifier
import datetime
import math
import pickle
from scipy.optimize import curve_fit
from scipy import signal
from scipy.stats import spearmanr
import metpy.calc as mpcalc
from tensorflow.keras.models import model_from_json
import random
import scipy.stats as sstat
from sklearn.metrics import roc_auc_score, roc_curve
from datetime import datetime, timedelta
import matplotlib.dates as mdates

#=============================================================================
#develop and evaluate model2: the XGBoost model and evalaute it to produce Fig2(g-h)
#=============================================================================
def get_day_of_year(year, month, day):
    """
    Returns the day of the year for a given date.

    Args:
        year (int): The year.
        month (int): The month (1-12).
        day (int): The day of the month.

    Returns:
        int: The day of the year (1-indexed).
    """
    dt_object = datetime.date(year, month, day)
    return dt_object.timetuple().tm_yday

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

def harmonic_func(x,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10):
    pid=365
    ofreq = 2*math.pi/pid
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
    pid=365
    ofreq = 2*math.pi/pid
    params = [B0,B1,B2,B3,B4,B5,B6]
    funcomp = [1,
               np.cos(ofreq*x),np.sin(ofreq*x),
               np.cos(2*ofreq*x),np.sin(2*ofreq*x),
               np.cos(3*ofreq*x),np.sin(3*ofreq*x)               
               ]
    return np.sum(np.array(params)*np.array(funcomp))

def harmonic_func2(x,B7,B8,B9,B10):
    pid=365
    ofreq = 2*math.pi/pid
    params = [B7,B8,B9,B10]
    funcomp = [np.cos(4*ofreq*x),np.sin(4*ofreq*x), 
               np.cos(5*ofreq*x),np.sin(5*ofreq*x)            
               ]
    return np.sum(np.array(params)*np.array(funcomp))

def getNN(fd):
    json_file = open(fd+'.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(fd+'.weights.h5')
    return loaded_model

def lead_lag_corr2(xs,ys,tu):

    dict_shift=dict()
    dict_shift_p=dict()
    for dt in np.arange(1,tu+1,1): 
      cr=sstat.pearsonr(xs[dt:],ys[:-dt])
      dict_shift[dt]=cr #amv leads
    
      #dict_shift[0]=np.corrcoef(xs,ys)[0,1]
      dict_shift[0]=sstat.pearsonr(xs,ys)


      cr=sstat.pearsonr(xs[:-dt],ys[dt:])
      dict_shift[-dt]=cr #amv lags
      #break
    
    dtC=np.arange(-tu,tu+1,1)
    
    crC=[]
    for dt in dtC:
      crC.append(dict_shift[dt])
    
    crC=np.array(crC) 

    return crC,dtC

def AMcond():
    m1 = 6
    m2 = 8
    datH='/Users/yiwenmao/Documents/StaFront2/biau/'
    fn = datH+'mb850/q_'+str(m1)+'_'+str(m2)+'.mat'
    MC=sio.loadmat(fn)
    q850 = MC['data']*1000

    fn = datH+'mb850/ept_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    ept850 = MC['data']

    fn = datH+'mb850/u_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    u850 = MC['data']   

    fn = datH+'mb850/v_'+str(m1)+'_'+str(m2)+'.mat'
    MC = sio.loadmat(fn)
    v850 = MC['data']   

    fn = datH+'mb850/dField2C_dy.mat'
    MC = sio.loadmat(fn)
    dept = MC['ept']  
    dq = MC['q']  
    du = MC['u']
    dv = MC['v']
    return q850,ept850,dept,dq,u850,v850,du,dv

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
#========================================================================
fn = 'data/ERA5_JJA_pred.mat'
MC = sio.loadmat(fn)
lonn=MC['lon'].flatten()
latn=MC['lat'].flatten()
dtLC=MC['datetimeList']

fn = 'data/bFt.mat'
MC = sio.loadmat(fn)
XdataFt = MC['HarmonicJJA'] #predictors based on harmonic decmopsion of vorticiyt and MFC
FtL = MC['FtL']
Ydata = MC['isbFt']
ftobs = MC['ftobs'].flatten()
#datH='/Users/yiwenmao/Documents/StaFront2/biau/'
#bn,latn,lonn,dtLC,Yft,isft,Ydata=getbn()
Nt,_ = Ydata.shape
#dx, dy = mpcalc.lat_lon_grid_deltas(lonn, latn)
#dx = dx/1000
#dy = dy/1000

Xdata = np.concatenate((XdataFt,FtL),axis=1) #FtL is the length of predicted fronts
_,Nf = Xdata.shape

#modle training
#select the parameters 
lr = 0.01 
maxd = 5 
nest = 500
Fmx = np.zeros((21,Nf))
Ypred=[]
Yprob=[]
for yr,k in zip(np.arange(2000,2021),np.arange(0,21)):

    testInd=dtLC[:,0]==yr
    trainInd=~testInd
    Xtrain=Xdata[trainInd,:]
    Xtest=Xdata[testInd,:]
    Ytrain=Ydata[trainInd,:]
    Ytest=Ydata[testInd,:]

    clf = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=nest,       # number of trees
    max_depth=maxd,           # depth of each tree
    learning_rate=lr     # shrinkage rate
    )

    clf.fit(Xtrain, Ytrain)
    Fmx[k,:]=clf.feature_importances_
    # Make predictions on the test set
    y_pred = clf.predict(Xtest)
    y_pred = y_pred==1
    y_pred = y_pred.astype(float)
    Ypred.append(y_pred)
    prob = clf.predict_proba(Xtest)
    Yprob.append(prob)
    ACC,POD,POFD,FAR,SR,CSI,bias=getMet(1,Ytest[:,1],y_pred[:,1])
    
    '''
    if CSI>=0.6:
        print(yr)
        print([ACC,CSI,POD])

        model_file_path = 'var_model/xgboost_isbiau_from_bnpred_test3_'+str(yr)+'.pkl' #store models with relatively good skills (they will be applied to d4PDF later)
        with open(model_file_path,'wb') as model_file:
            pickle.dump(clf,model_file)
            print(model_file)   
    '''
Ypred = np.concatenate(Ypred[:],axis=0)
Ypred = Ypred.astype(float) 
Yprob = np.concatenate(Yprob[:],axis=0)
ACCo,PODo,POFDo,FARo,SRo,CSIo,biaso=getMet(1,Ydata[:,1],Ypred[:,1])

mdl =  xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=nest,       # number of trees
    max_depth=maxd,           # depth of each tree
    learning_rate=lr,     # shrinkage rate
    )

mdl.fit(Xdata, Ydata)
#importance_scores = mdl.get_booster().get_score(importance_type='gain')
#importance_scores1 = mdl.get_booster().get_score(importance_type='weight')
#importance_scores2 = mdl.get_booster().get_score(importance_type='cover')

#model evaluation
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
f1 = f1_score(Ydata[:,1], Ypred[:,1])
kappa = cohen_kappa_score(Ydata[:,1], Ypred[:,1])
mcc = matthews_corrcoef(Ydata[:,1], Ypred[:,1])


#model_file_path = 'var_model/xgboost_isbiau_from_bnpred_test3_all.pkl'
#with open(model_file_path,'wb') as model_file:
#    pickle.dump(mdl,model_file)
#    print(model_file)

dfout = pd.DataFrame(index=np.arange(0.05,1,0.05),columns=['ACC','POD','POFD','FAR','SR','CSI','bias'])
for p in np.arange(0.05,1,0.05):
    ACC,POD,POFD,FAR,SR,CSI,bias = getMet(1,Ydata[:,1],Yprob[:,1]>=p)
    dfout.loc[p,'ACC'] = ACC
    dfout.loc[p,'POD'] = POD
    dfout.loc[p,'POFD'] = POFD
    dfout.loc[p,'FAR'] = FAR
    dfout.loc[p,'SR'] = SR
    dfout.loc[p,'CSI']=CSI
    dfout.loc[p,'bias']=bias

#ACC,POD,POFD,FAR,SR,CSI,bias = getMet(True,Ydata,Ypred)
fpr, tpr, thresholds = roc_curve(Ydata[:,1], Yprob[:,1])
auc = roc_auc_score(Ydata[:,1], Yprob[:,1])

#see https://scikit-learn.org/1.1/auto_examples/model_selection/plot_roc.html
fig,ax=plt.subplots(1,1,figsize=(5,5))
ax.plot(fpr,tpr,label="AUC = %1.2f" % auc)
xs = np.arange(0,1.1,0.1)
ys = xs
ax.plot(xs,ys,'r--',label='1:1 (random predction)')
ax.legend()
ax.set_xlabel('False Positive rate',fontsize=16)
ax.set_ylabel('True Postive rate',fontsize=16)
ax.set_title('Receiver Operating Characteristic (ROC)',fontsize=14)

dsROC=pd.DataFrame({'FPR':fpr,'TPR':tpr})
fout='ROC.csv'
dsROC.to_csv(fout)

#calculate frequency of biau fronts by prediction using Ypred here

JJA_dayEY = np.arange(152,244)
dtList = dtLC
yrC = np.unique(dtList[:,0])
Nyr = len(yrC)
Nssn = 92
cmx = np.zeros((Nyr,Nssn))
for yi in np.arange(0,Nyr):
    cmx[yi,:] = JJA_dayEY
JJA_dayY = cmx.flatten()

PredFt = np.zeros(JJA_dayEY.shape)
for i in np.arange(0,len(JJA_dayEY)):
    ck = JJA_dayY==JJA_dayEY[i]
    PredFt[i]=Ypred[ck,1].sum()/Nyr

#fout = 'test_data/PredBiau_ref.mat'
#sio.savemat(fout,{'PredFt':PredFt,'JJA_dayEY':JJA_dayEY})
#print(fout)

RMSE=np.sqrt(np.sum((PredFt - ftobs)**2)/len(PredFt))

#plt.plot(ftobs);plt.plot(PredFt)
start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)

fig,axs = plt.subplots(1,1,figsize=(6,5))
axs.plot(dates,ftobs,'co-',label='Observation')
axs.plot(dates,PredFt,'mo-',label='Prediction')
axs.set_xlim((dates[0],dates[-1]))
axs.legend()
axs.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
axs.set_ylabel('Baiu Front Freq (day)',fontsize=16)
axs.set_xlabel('JJA dates',fontsize=16)
