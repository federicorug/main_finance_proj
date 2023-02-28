import numpy as np 
from scipy.stats import norm

r=0.1
s=30
k=40
t=240/365
sigma=0.3

def BandS(r,s,k,t,sigma,type="C"):
    d_1=(np.log(s/k)+t*(r+0.5*sigma**2))/(sigma*np.sqrt(t))
    d_2=(np.log(s/k)+t*(r-0.5*sigma**2))/(sigma*np.sqrt(t))
    if type=="c":
        bns=s*norm.cdf(d_1,0,1)-k*np.exp(-r*t)*norm.cdf(d_2,0,1)
    if type=="p":
        bns=k*np.exp(-r*t)*norm.cdf((-d_2),0,1)-s*norm.cdf((-d_1),0,1)
    return bns

print(BandS(r,s,k,t,sigma,type="c"))
