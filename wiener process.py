#wiener process
import numpy as np 
import matplotlib.pyplot as plt 
path=1000
day=252
dt=np.linspace(0, 1,num=day,retstep=1)
dw=np.ones(day)
wt_cum=np.zeros((day,path))
index2=0
for i in range(path):
    index=0
    for t in dt[0]:
        wt_cum[index,i]=np.sqrt(dt[1])*np.random.randn()
        index+=1
plt.plot(wt_cum.cumsum(axis=0))
#plt.hist(wt_cum[7,],density=1,bins=100)
plt.show()
   
