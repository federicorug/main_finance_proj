import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad
import os
from datetime import datetime

def bns(
      s, k, r, tau, sigma, opt="C"):

    "calcolo il prezzo BnS, delta=Nd1 per call o put options"

    d1 = (np.log(np.array(s)/k) + (r+sigma*sigma/2) * np.array(tau))/ (sigma
                                                    *np.sqrt(np.array(tau)))
    d2 = d1 - sigma * np.sqrt(tau)

    if opt == "C":
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2, 0, 1)
        price = Nd1 * s - Nd2 * k * np.exp(- r * tau)
        delta = Nd1

    if opt == "P":
        Nd1 = - norm.cdf(- d1)
        Nd2 = norm.cdf(- d2)
        price = Nd2 * k * np.exp(- r * tau) + Nd1 * s
        delta = Nd1
        
    return price, delta

def integ(
      fi, r, b, u, a, rho, tau, k, x, volofvol, sigma):

    i = complex(0, 1)
    d = np.sqrt(((rho * volofvol * fi * i - b) 
                 ** 2)- (volofvol ** 2) * (2 * u * fi * i - fi ** 2))
    g = (b - rho * volofvol * fi * i + d) / (b - rho * volofvol * fi * i - d)
    D = ((b - rho * volofvol * fi * i + d) /
         (volofvol ** 2))* ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    C = r * fi * i * tau + (a / volofvol ** 2)* ((b - rho * volofvol *
            fi * i + d) * tau- 2 *np.log((1 - g * np.exp(d * tau)) / (1 - g)))
    phi = np.exp(C + D * sigma + i * fi * x)
    re = np.real((np.exp(- i * fi * np.log(k)) * phi) / (i * fi))


    return re

def heston(
      s, k, r, tau, sigma, lamda, rho, lr_var, volofvol, mean_rev, opt = "C"):

    'calcolo il prezzo e il delta di opzioni call o put usando la funzione caratteristica dell heston model'

    x = np.log(s)
    b1 = mean_rev + lamda - rho * volofvol
    b2 = mean_rev + lamda
    u1 = 0.5
    u2 = - 0.5
    a = mean_rev * lr_var
    p1, _ = quad(integ, 0, 100, args = (r, b1, u1, a, rho, tau, k,
                                        x, volofvol, sigma))
    p2, _ = quad(integ, 0, 100, args = (r, b2, u2, a, rho, tau, k,
                                        x, volofvol, sigma))
    p1 = 0.5 + (1 / np.pi) * p1
    p2 = 0.5 + (1 / np.pi) * p2


    if opt == 'C':

        price=s*p1-k*np.exp(-r*tau)*p2
        delta = p1

    else:

        price=k*np.exp(-r*tau)*(1-p2)-s*(1-p1)
        delta = -(1 - p1)

    return price, delta


class dataseries():

    "this class collect data and split the dataframe in a series of date and prices"

    def __init__(
              self, dataframe_dir):

        self.snp = pd.read_csv(dataframe_dir)  #----------------------------------- csv with the dataframe
        self.snp['Price'] = self.snp['Price'].str.replace(',', '').astype(float)
        self.price_series = self.snp['Price'][: : - 1]#---------------------------- the time series of prices
        self.date_series =  pd.to_datetime(self.snp['Date'])#---------------------- create the date series


class option(dataseries):
    
    "this class collect main info of a specific option and cut the price series till the maturity date"

    def __init__(
       self, dataframe_pos, k, maturity,  opt):

        super().__init__(dataframe_pos)

        self.timematurity = datetime.strptime(maturity, '%m/%d/%Y')#--------------- option maturity in days
        self.day_to_m=maturity#---------------------------------------------------- maturity day in time delta
        self.maturity = round((((self.timematurity
                               - self.date_series.iloc[- 1]).days)) / 360,6)#------ time to maturity
        self.steps =len(self.date_series) - int(
            self.date_series[self.date_series == self.timematurity].index[0])#----- timesteps to discretize time
        self.time = np.linspace(0, self.maturity, self.steps)#--------------------- discretization of time
        self.tau = self.time[ : : - 1]#-------------------------------------------- discounting time
        self.tau = self.tau[ : self.steps]#---------------------------------------- time to maturity series
        self.k = k#---------------------------------------------------------------- strike price
        self.opt = opt#------------------------------------------------------------ option type call or put
        self.price_series = self.price_series.iloc[ : self.steps]#----------------- cut the price series according to maturity


class opt_model(option):
    
    'this class is initialized by defining an option, its methods return the price and the delta series according to black or heston'

    def __init__(
            self, dataframe_pos, k, maturity,  opt):

        super().__init__(dataframe_pos, k, maturity,  opt)

    def Black(self, r, sigma):

        self.r = r#---------------------------------------------------------------- risk free rate
        self.sigma = sigma#-------------------------------------------------------- constant volatility
        self.price_list, self.delta =bns(
            self.price_series , self.k, self.r, self.tau, self.sigma, self.opt)

        return self.price_list, self.delta

    def Heston(self, r, lamda, rho, lr_var, volofvol, mean_rev,sigma):

        self.r = r#---------------------------------------------------------------- risk free rate
        self.lamda = lamda#-------------------------------------------------------- volatility risk premium
        self.rho = rho#------------------------------------------------------------ correlation vol\underlyng
        self.lr_var = lr_var#------------------------------------------------------ long run variance
        self.volofvol = volofvol#-------------------------------------------------- volatility of volatility
        self.mean_rev = mean_rev#-------------------------------------------------- mean reversion rate of the volatility
        self.sigma = sigma#-------------------------------------------------------- volatility in time zero
        self.price_list = []
        self.delta_list = []

        for i in range(0, len(self.price_series)):

            self.price, self.delta = heston(np.array(self.price_series)[i], self.k, self.r,
                                            self.tau[i], self.sigma, self.lamda, self.rho,
                                            self.lr_var, self.volofvol, self.mean_rev, self.opt)
            self.price_list.append(self.price)
            self.delta_list.append(self.delta)

        return self.price_list, self.delta_list



class signal(opt_model):
    
    'this class create, according to the method, a vector of true and false for all the discretized timeline'
    'true is the moment the strategy close the delta'

    def __init__(
            self, dataframe_pos, k, maturity,  opt):

        super().__init__(
             dataframe_pos, k, maturity,  opt)


    def time_rebalance(
            self, rebalance_days):

        self.rebalance_days = rebalance_days#-------------------------------------- day to rebalance
        self.trade_signal = [False for i in range(len(self.price_series))]#-------- initialize signal vector
        
        for i in range(0,len(self.price_series),self.rebalance_days):
            
           self.trade_signal[i] = True
        
        return self.trade_signal
        

class strategy(signal):

    def __init__(
            self, dataframe_pos, k, maturity,  opt):

        super().__init__(dataframe_pos, k, maturity,  opt)

        
    def B_time_trade(
            self,  r, sigma, rebalance_days):
        
        self.sigma = sigma
        self.r = r
        self.time_rebalance(rebalance_days)#--------------------------------------- call the method and obtain the signal vector
        self.price, self.delta = self.Black( self.r, self.sigma)#------------------ option price and delta usins black
        self.opt_pnl = []#--------------------------------------------------------- option price series with black and sholes
        self.delta_pnl = []#------------------------------------------------------- series of option delta 
        self.temp_price = self.price.iloc[0]#-------------------------------------- temporary variable for the price of the option
        self.temp_delta = self.delta[0]#------------------------------------------- temporary variable for the delta of the option
        self.temp_und = self.price_series.iloc[0]#--------------------------------- temporary variable for the value of the underlying
        self.strat_pnl = []#------------------------------------------------------- list for the strategy's pnl initialization
        count=0
        
        'modello: black and scholes'
        'per i nella serie dei prezzi, se il segnale è uguale a True:'
        'fai la differenza tra il valore dell opzione al tempo t e t-1----> quanto è variato il prezzo dell opzione'
        'fai la differenza tra il sottostante in t e t-1 e moltiplica per il delta dell opz in t-1------> variazione di pnl della parte allocata in asset'
        'se l opzione è una call e il sottostante è salito o una put e il sottostante è sceso:'
        'il pnl è dato dalla variazione dell opzione meno quello dell sottostante'
        'in altro caso:'
        'il pnl è dato dalla variazione dell sottostante meno quella dell opzione'
        
        for i in range(1, len(self.price_series)):
            if  self.trade_signal[i] == True:
                
                self.opt_pnl.append( self.price.iloc[i] - self.temp_price )
                self.delta_pnl.append( (self.price_series.iloc[0] -
                                        self.temp_und) * self.temp_delta )
                
                if self.opt == "C" and self.price_series.iloc[i] >= self.temp_und or self.opt == "P" and self.price_series.iloc[i] <= self.temp_und  :
                    
                    self.strat_pnl.append( self.opt_pnl[count] - self.delta_pnl[count])
                else:
                    self.strat_pnl.append(self.delta_pnl[count] - self.opt_pnl[count] )

                self.temp_price = self.price.iloc[i]
                self.temp_delta = self.delta[i]
                self.temp_und = self.price_series.iloc[i]
                count += 1
        
        return self.opt_pnl, self.delta_pnl, self.strat_pnl
    
    def H_time_trade(
            self,  r, lamda, rho, lr_var, volofvol, mean_rev, sigma, rebalance_days):
        
        self.r = r
        self.lamda = lamda
        self.rho = rho
        self.lr_var = lr_var
        self.volofvol = volofvol
        self.mean_rev = mean_rev
        self.sigma = sigma
        self.time_rebalance(rebalance_days)
        self.price, self.delta = self.Heston(  self.r, self.lamda, self.rho,
                                self.lr_var, self.volofvol, self.mean_rev,sigma)
        self.opt_pnl = []
        self.delta_pnl = []
        self.temp_price = self.price[0]
        self.temp_delta = self.delta[0]
        self.temp_und = self.price_series.iloc[0]
        self.strat_pnl = []
        count=0
        
        'modello: heston'
        
        for i in range(1, len(self.price_series)):
            if  self.trade_signal[i] == True:
                
                self.opt_pnl.append( self.price[i] - self.temp_price )
                self.delta_pnl.append( (self.price_series.iloc[0] -
                                        self.temp_und) * self.temp_delta )
                
                if self.opt == "C" and self.price_series.iloc[i] >= self.temp_und or self.opt == "P" and self.price_series.iloc[i] <= self.temp_und  :
                    
                    self.strat_pnl.append( self.opt_pnl[count] - self.delta_pnl[count])
                else:
                    self.strat_pnl.append(self.delta_pnl[count] - self.opt_pnl[count] )
                

                self.temp_price = self.price[i]
                self.temp_delta = self.delta[i]
                self.temp_und = self.price_series.iloc[i]
                count += 1
            
        return self.opt_pnl, self.delta_pnl, self.strat_pnl
                
  

os.chdir(r"C:\Users\j058276\Desktop\python\data")#--------------------------------- select directory 
snp_pos="S&P 500 Historical Data (1) (3).csv"#------------------------------------- select file 

r=0.05
k=3850
sigma=0.22
maturity="02/14/2024"

v=0.12#---------------------------------------------------------------------------- volofvol
lamda=0.01#------------------------------------------------------------------------ variance rp
rho=-0.57#------------------------------------------------------------------------- correlation
lr_var=0.22#----------------------------------------------------------------------- long run var
mean_rev=0.1#---------------------------------------------------------------------- mean rev
 
strat1=strategy(snp_pos,k, maturity, 'P')
a,b,c = strat1.H_time_trade(r, lamda, rho, lr_var, v, mean_rev, sigma, 10)
print(np.sum(c))

strat2=strategy(snp_pos,k, maturity, 'P')
a1,b1,c1 = strat2.B_time_trade(r, sigma, 10)
print(np.sum(c1))



