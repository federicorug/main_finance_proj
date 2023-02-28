#montecarlo sim for ptf
import numpy as np 
import scipy.stats as sc
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt

def gettin_data(stock_list, startdate,endate):
    raw_data=pdr.get_data_yahoo(stock_list,startdate,endate)
    raw_data=raw_data['Close'].dropna(axis=1)
    ret=np.diff(np.log(raw_data),axis=0)
    meanret=np.mean(ret,axis=0)
    cov_mat=np.cov(ret,rowvar=False)
    return meanret,cov_mat,ret


stock_list=['AMZN','ESLOF','MSFT','GOOG']
endate=dt.datetime.now()
startdate=endate-dt.timedelta(300)
sim_n=300
time=100
meanret,cov_mat,ret=gettin_data(stock_list, startdate,endate)
meanm=np.full((time,len(stock_list)), fill_value=meanret)
ptf_sim=np.full((time,sim_n), fill_value=0)
weight=np.array((0.25,0.25,0.25,0.25))
weight=np.atleast_2d(weight).T
initial_vaule=100000


for i in range(0, sim_n):
    z=np.random.normal(size=(time,len(stock_list)))
    l=np.linalg.cholesky(cov_mat)
    a=np.inner(z,l)
    daily_ret=meanm+a
    ptf_ret=np.cumprod((np.dot(daily_ret,weight)+1))*initial_vaule
    ptf_ret=np.atleast_2d(ptf_ret).T
    ptf_sim[:,i]=ptf_ret[:,0]

plt.plot(ptf_sim)
plt.show()