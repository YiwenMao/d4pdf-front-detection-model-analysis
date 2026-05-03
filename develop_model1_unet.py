import numpy as np
import scipy.io as sio
import pandas as pd
#from datetime import datetime, timedelta
from netCDF4 import Dataset
from metpy.units import units
import metpy.calc as mpcalc
from scipy.ndimage import gaussian_filter
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import regularizers
from keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Reshape, Dense, Multiply
import sys
import math
#=====================================================================================================================
#This is a script used to develope model 1 (U-NET)

def getMet(cc,ytest,ypred):
    OT=ytest==cc
    PT=ypred==cc
    OF=ytest!=cc
    PF=ypred!=cc
    A=(OT&PT).sum()
    B=(PT&OF).sum()c
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

#define the U-NET structure
from tensorflow.keras import layers, models
def unet_model(input_shape, num_classes=2,nft=32,l2_reg=0.0001):
    inputs = layers.Input(shape=input_shape)
    reg = regularizers.l2(l2_reg)

    # Encoder (Downsampling)
    conv1 = layers.Conv2D(nft, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(inputs)
    conv1 = layers.Conv2D(nft, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)  # Down to (6, 6)

    conv2 = layers.Conv2D(nft*2, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(pool1)
    conv2 = layers.Conv2D(nft*2, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)  # Down to (3, 3)

    # Bottleneck
    conv3 = layers.Conv2D(nft*4, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(pool2)
    conv3 = layers.Conv2D(nft*4, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(conv3)

    #Dropout(0.2)
    # Decoder (Upsampling)
    up1 = layers.Conv2DTranspose(nft*2, (2, 2), strides=(2, 2), padding='same')(conv3)
    concat1 = layers.Concatenate()([up1, conv2])
    conv4 = layers.Conv2D(nft*2, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(concat1)
    conv4 = layers.Conv2D(nft*2, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(conv4)

    up2 = layers.Conv2DTranspose(nft, (2, 2), strides=(2, 2), padding='same')(conv4)
    concat2 = layers.Concatenate()([up2, conv1])
    conv5 = layers.Conv2D(nft, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(concat2)
    conv5 = layers.Conv2D(nft, (3, 3), activation='relu', padding='same',kernel_regularizer=reg)(conv5)

    # Attention gate
    dense = Dense(1, activation='sigmoid')(conv5)
    attention = Reshape((24, 24, 1))(dense)
    attention = Multiply()([conv5, attention])

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(attention)  # For binary segmentation

    model = models.Model(inputs, outputs)
    return model

#define MFC
def getMoist(qu,qv,dy,dx):
    Nt,nd1,nd2 = qu.shape
    dqudx = np.zeros((Nt,nd1,nd2))
    dqvdy = np.zeros((Nt,nd1,nd2))
    for ti in np.arange(0,Nt):
        dqudx[ti,:,:],_ = mpcalc.gradient(qu[ti,:,:],deltas=(dy,dx))
        _,dqvdy[ti,:,:] = mpcalc.gradient(qv[ti,:,:],deltas=(dy,dx))
    dMoist = -1*(dqudx+dqvdy)
    return dMoist

#find the length of a front
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
#=======================================================
#datH='/Users/yiwenmao/Documents/StaFront2/biau/'
datH='/external12/yiwen/StaFront2/biauFront/'
m1=6
m2=8
fn = datH+'stcoord_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
latn = MC['lat'].flatten()
lonn = MC['lon'].flatten()
dx, dy = mpcalc.lat_lon_grid_deltas(lonn, latn)
dx = dx/1000
dy = dy/1000

#load the gridded front data (the predictand)
fn = datH+'GridFront_'+str(m1)+'_'+str(m2)+'.mat'
MC = sio.loadmat(fn)
Yft=MC['Yft']
dtLC=MC['dtLC']
Nt,nd1,nd2 = Yft.shape
for i in np.arange(0,Nt):
    Yft[i,:,:]=np.flipud(Yft[i,:,:])


Ydata = np.zeros((Nt,nd1,nd2,2),dtype=bool)
Ydata[:,:,:,0] = True
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

for i in np.arange(0,Nt):
    if isft[i]==True:
        Ydata[i,:,:,0]= Yft[i,:,:]==0 #no Front
        Ydata[i,:,:,1]= Yft[i,:,:]==1      
#-------------------------------------------------
#load the ERA5 data for predictors
fn = datH+'mb925/q_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
q925 = MC['data']*1000 #specific humidity at 925 mb

fn = datH+'mb850/q_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
q850 = MC['data']*1000 #specific humidity at 850 mb

fn = datH+'mb700/q_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
q700 = MC['data']*1000 #specific humidity at 700mb

#------------------------------------------------
fn = datH+'mb925/u_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
u925 = MC['data']

fn = datH+'mb850/u_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
u850 = MC['data']

fn = datH+'mb700/u_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
u700 = MC['data']
#------------------------------------------------
fn = datH+'mb925/v_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
v925 = MC['data']

fn = datH+'mb850/v_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
v850 = MC['data']

fn = datH+'mb700/v_'+str(m1)+'_'+str(m2)+'.mat'
MC=sio.loadmat(fn)
v700 = MC['data']
#-----------------------------------------------
fn = datH+'mb700/dField2C_dy.mat'
dField2C_700 = sio.loadmat(fn)
dw700_dy = dField2C_700['w']

fn = datH+'mb850/dField2C_dy.mat'
dField2C_850 = sio.loadmat(fn)
dw850_dy = dField2C_850['w']

fn = datH+'mb925/dField2C_dy.mat'
dField2C_925 = sio.loadmat(fn)
dw925_dy = dField2C_925['w']
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn = datH+'mb700/dField2C_dy.mat'
dField2C_700 = sio.loadmat(fn)
dept700_dy = dField2C_700['ept']

fn = datH+'mb850/dField2C_dy.mat'
dField2C_850 = sio.loadmat(fn)
dept850_dy = dField2C_850['ept']

fn = datH+'mb925/dField2C_dy.mat'
dField2C_925 = sio.loadmat(fn)
dept925_dy = dField2C_925['ept']
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn = datH+'mb700/dField2C_dy.mat'
dField2C_700 = sio.loadmat(fn)
dvort700_dy = dField2C_700['vort']

fn = datH+'mb850/dField2C_dy.mat'
dField2C_850 = sio.loadmat(fn)
dvort850_dy = dField2C_850['vort']

fn = datH+'mb925/dField2C_dy.mat'
dField2C_925 = sio.loadmat(fn)
dvort925_dy = dField2C_925['vort']
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn = datH+'mb925/ObjFD.mat'
MC = sio.loadmat(fn)
eq3_925 = MC['eq3']
mask1_925 = MC['mask1']

fn = datH+'mb850/ObjFD.mat'
MC = sio.loadmat(fn)
eq3_850 = MC['eq3']
mask1_850 = MC['mask1']

fn = datH+'mb700/ObjFD.mat'
MC = sio.loadmat(fn)
eq3_700 = MC['eq3']
mask1_700 = MC['mask1']
#--------------------------------------------------

qu925 = q925*u925
qv925 = q925*v925
qu850 = q850*u850
qv850 = q850*v850
qu700 = q700*u700
qv700 = q700*v700

Xdata = np.zeros((Nt,nd1,nd2,18))
Xdata[:,:,:,0]=getMoist(qu925,qv925,dy,dx)
Xdata[:,:,:,1]=getMoist(qu850,qv850,dy,dx)
Xdata[:,:,:,2]=getMoist(qu700,qv700,dy,dx)
Xdata[:,:,:,3]=dw700_dy
Xdata[:,:,:,4]=dw850_dy
Xdata[:,:,:,5]=dw925_dy
Xdata[:,:,:,6]=dvort700_dy
Xdata[:,:,:,7]=dvort850_dy
Xdata[:,:,:,8]=dvort925_dy
Xdata[:,:,:,9]=dept700_dy
Xdata[:,:,:,10]=dept850_dy
Xdata[:,:,:,11]=dept925_dy
Xdata[:,:,:,12]=eq3_925
Xdata[:,:,:,13]=mask1_925
Xdata[:,:,:,14]=eq3_850
Xdata[:,:,:,15]=mask1_850
Xdata[:,:,:,16]=eq3_700
Xdata[:,:,:,17]=mask1_700

_,_,_,Np = Xdata.shape
for k in np.arange(0,Np):
    vmin = np.nanmin(Xdata[:,:,:,k])
    vmax = np.nanmax(Xdata[:,:,:,k])
    Xdata[:,:,:,k] = 2*(Xdata[:,:,:,k]-vmin)/(vmax-vmin)-1 

vnm = 'new3' #just a name to label the version of the model
testyr=int(sys.argv[1]) #year of testing from 2000 to 2021
#testyr=2010
#data preparation
tk = dtLC[:,0]==testyr
input_shape= (nd1,nd2,Np)
Xtest = Xdata[tk,:,:]
Xtrain = Xdata[~tk,:,:]
Ytest = Ydata[tk,:]
Ytrain = Ydata[~tk,:]

#model training
num_classes = 2
model = unet_model(input_shape)
model.summary()

earlystopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy', metrics=['accuracy'])
model.fit(Xtrain, Ytrain, epochs=500, batch_size=8, validation_data=(Xtest, Ytest),callbacks=[earlystopping],verbose=2)

Ypred = model.predict(Xtest)
y_pred_class = Ypred[:,:,:,1]>0.5
y_test_class = Ytest[:,:,:,1]

#evaluate the model
sigc=1
ACC,POD,POFD,FAR,SR,CSI,bias = getMet(sigc,y_test_class,y_pred_class)
dfr=pd.DataFrame(index=['CSI','POD','bias','SR','ACC'],columns = ['Ft','NoFt'])
dfr.loc['CSI','Ft'] = CSI
dfr.loc['POD','Ft'] = POD
dfr.loc['bias','Ft'] = bias
dfr.loc['SR','Ft'] = SR
dfr.loc['ACC','Ft'] = ACC
sigc=0
ACC,POD,POFD,FAR,SR,CSI,bias = getMet(sigc,y_test_class,y_pred_class)
dfr.loc['CSI','NoFt'] = CSI
dfr.loc['POD','NoFt'] = POD
dfr.loc['bias','NoFt'] = bias
dfr.loc['SR','NoFt'] = SR
dfr.loc['ACC','NoFt'] = ACC
print(dfr)

outH = 'unet/ftNft/'+vnm+'/'
model_json = model.to_json()
fout = outH+'build1_'+str(testyr)
with open(fout+'.json','w') as json_file:
    json_file.write(model_json)
model.save_weights(fout+'.weights.h5') 
print(fout)

