import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#======================================================
#calculate the the average number of days of Baiu fronts per month in JJA (Fig.3c)
#======================================================

fn = 'data/ERA5_JJA_pred.mat'
MC = sio.loadmat(fn)
dtLC = MC['datetimeList']
bn = MC['isbft'].flatten()
yrC = np.arange(2000,2021)
m6C = []
m7C = []
m8C = []
for yr in np.arange(2000,2021):
    ck = (dtLC[:,0]==yr)&(dtLC[:,1]==6)
    m6C.append(bn[ck].sum())
    ck = (dtLC[:,0]==yr)&(dtLC[:,1]==7)
    m7C.append(bn[ck].sum())
    ck = (dtLC[:,0]==yr)&(dtLC[:,1]==8)
    m8C.append(bn[ck].sum())
m6C = np.array(m6C)
m7C = np.array(m7C)
m8C = np.array(m8C)

cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/bft/'
fn = cH+'HPB_ts.mat'
MC = sio.loadmat(fn)
d6 = MC['ft6'].flatten()
d7 = MC['ft7'].flatten()
d8 = MC['ft8'].flatten()
count_bft=d6+d7+d8
data = [d6,d7,d8]
dataO = [m6C,m7C,m8C]

cH='/Users/yiwenmao/Documents/StaFront3/result/YearData/bft/'
fn = cH+'HPB_ts.mat'
MC = sio.loadmat(fn)
d6 = MC['ft6'].flatten()
d7 = MC['ft7'].flatten()
d8 = MC['ft8'].flatten()
count_nbft=d6+d7+d8

# Your data
data_d = [d6, d7, d8]
data_m = [m6C, m7C, m8C]

# Positions for groups
positions_d = [1, 2, 3]
positions_m = [p + 0.35 for p in positions_d]  # shift right

# Plot boxplots
plt.boxplot(data_d, positions=positions_d, widths=0.3, patch_artist=False)
#plt.boxplot(data_m, positions=positions_m, widths=0.3, patch_artist=True)

# X-axis labels centered between pairs
plt.xticks([p for p in positions_d], ['Jun', 'Jul', 'Aug'])

plt.show()
