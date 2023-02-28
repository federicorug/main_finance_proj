
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sigma=0.25
s0=100
mu=0.050
t=1
path=10
obs=t*252
dt=np.transpose(np.atleast_2d(np.linspace(0, t, obs)))
x=np.random.randn(obs,path)
Wt=np.sqrt(dt)*x
st=s0*np.exp((mu-0.5*sigma**2)*dt+sigma*Wt)

df=pd.DataFrame(st)

n_raws,n_col=st.shape
col_name=["path "+str(i) for i in range(n_col) ]
df.columns=col_name
df=df.T
raw_name=["time "+str(i) for i in range(n_raws) ]
df.columns=raw_name
df=df.T
df["col"]=5
df.drop(labels=["col"], axis=1)
print(df)
print(dt)
plt.plot(dt.T,df[:,:])




