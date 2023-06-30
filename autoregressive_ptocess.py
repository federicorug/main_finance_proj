#autoregressive_process
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
def ARprocess(s0,years,mu,beta,sigma=2.5,n_obs=252):
    ts=np.linspace(0,years,n_obs)
    s=np.zeros_like(ts)
    s[0]=s0
    for i,t in enumerate(ts[1:],start=1):
        s[i]=mu + beta*s[i-1]+np.sqrt(sigma)*np.random.randn()
    return s

def MApath(s0,years,mu,beta,sigma=2.5,n_obs=252):
    ts=np.linspace(0,years,n_obs)
    s=np.zeros_like(ts)
    s[0]=s0
    errors_mat=np.zeros_like(ts)
    errors_mat[0]=np.random.randn()
    for i,t in enumerate(ts[1:],start=1):
        errors_mat[i]=np.random.randn()
        s[i]=mu + beta*errors_mat[i-1]+np.sqrt(sigma)*np.random.randn()
    return s

def autoreg_ma(s0,years,mu,beta,theta,sigma=2.5,n_obs=252):
    ts=np.linspace(0,years,n_obs)
    s=np.zeros_like(ts)
    s[0]=s0
    errors_mat=np.zeros_like(ts)
    errors_mat[0]=np.random.randn()
    for i,t in enumerate(ts[1:],start=1):
        errors_mat[i]=np.random.randn()
        s[i]=mu + beta*errors_mat[i-1]+theta*s[i-1]+np.sqrt(sigma)*np.random.randn()
    return s
stak=autoreg_ma(100, 1, 1, 0.99, 0.9)
stack2=autoreg_ma(100, 1, 1, 0.9, 0.9,25)

df=pd.DataFrame({"stack":stak,'stack2':stack2})
print(df)
