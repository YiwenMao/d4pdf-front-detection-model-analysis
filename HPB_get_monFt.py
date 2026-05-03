import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#====================================================================
#get the tempora- spatial datasets for baiu front/non-baiu fronts in June, July and August
#the data will be used later in the section of results
#====================================================================
#--------------------------------------------------------------------
#get the date index in JJA in a year
start = datetime(2026,6,1)
end = datetime(2026,8,31)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(days=1)
dates=np.array(dates)

Nssn = len(dates)
dssn = np.zeros((Nssn,2))
for di in np.arange(0,Nssn):
    dssn[di,0]=dates[di].month
    dssn[di,1]=dates[di].day
ind6 = dssn[:,0]==6
ind7 = dssn[:,0]==7
ind8 = dssn[:,0]==8
#--------------------------------------------------------------------
vnm ='bft' #choose either bft (baiu front) and nbft (non-baiu front)
cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/'
Nall=3068
Nd=24
dat6C = np.zeros((Nall,Nd,Nd))
dat7C = np.zeros((Nall,Nd,Nd))
dat8C = np.zeros((Nall,Nd,Nd))
isft6 = np.zeros((Nall,))
isft7 = np.zeros((Nall,))
isft8 = np.zeros((Nall,))
for i in np.arange(0,Nall):
    fn = cH+'HPB/yr'+str(int(i))+'.mat'
    MC = sio.loadmat(fn)
    data = MC[vnm]
    dat6C[i,:,:]=np.sum(data[ind6,:,:],axis=0)
    dat7C[i,:,:]=np.sum(data[ind7,:,:],axis=0)
    dat8C[i,:,:]=np.sum(data[ind8,:,:],axis=0)

    data1 = np.reshape(data,(92,Nd*Nd))
    isft = np.sum(data1,axis=1)>0
    isft6[i] = isft[ind6].sum()
    isft7[i] = isft[ind7].sum()
    isft8[i] = isft[ind8].sum()
    
    #break

fout = cH+vnm+'/HPB'+'_ts.mat'
sio.savemat(fout,{'dat6C':dat6C,'dat7C':dat7C,'dat8C':dat8C,
                  'ft6':isft6,'ft7':isft7,'ft8':isft8})
print(fout)