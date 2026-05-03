import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#====================================================================
#this is a script to calculate the relative frequency for 10000 bootstrapped samples for various HPB scenarios
# the generated data will later be used further analysis as shown in  Fig.4
#====================================================================
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

vnm ='bft'
cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/'
Nall = 3068
indp = np.arange(0,Nall)
Nd=24

isftC = np.zeros((92,Nall),dtype=bool)
for i in np.arange(0,Nall):
    fn = cH+'HPB/yr'+str(int(i))+'.mat'
    MC = sio.loadmat(fn)
    data = np.reshape(MC[vnm],(92,24*24))
    isftC[:,i] = np.sum(data,axis=1)>0

Nk = 10000
inda = np.arange(0,Nall)
arrC=np.zeros((92,Nk))
for i in np.arange(0,Nk):
    selected = np.random.choice(inda,size=Nall,replace=True)
    sub_isftC=isftC[:,selected]
    arrC[:,i] = np.sum(sub_isftC,axis=1)/len(selected)
    #break    

fout = 'result/YearData/'+vnm+'_HPB_time.mat'
sio.savemat(fout,{'data':arrC})
print(fout)  