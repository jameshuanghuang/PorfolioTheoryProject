#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:42:40 2020

@author: jameshuang
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#get data from csv file
data_stock = pd.read_csv('stock_prices_60.csv')

#interpolate missing data
for i in data_stock:
    data_stock[i]=data_stock[i].interpolate(method='bfill',limit_direction='backward',limit_area='outside')
 
    

#setting datetime index
data_stock['Date'] = pd.to_datetime(data_stock['Date'])
data_stock = data_stock.set_index('Date')




#daily
dt_daily_return = data_stock.pct_change().dropna()
data_daily_price = data_stock


import Backtest

wd = 12*2
dt_daily_return =  dt_daily_return.iloc[:,:]
data_daily_price = data_daily_price.iloc[:,:]

B = Backtest.Backtest(wd,dt_daily_return,data_daily_price)  

p = 20 #PCs
m = "rbf" # or poly
gma = 0.03#or poly degree

B.update_para(m,p,gma)
#Roll to get weight 
B.Roll()
#Initialize starting CASH and simulate portfolio
CASH = 10000
B.portfolio_val(CASH)
#Plot portfolio vs benchmark 

B.plot_portfolio_val()

#Plot tracking error
B.track_error()

#output table
#B.to_excel()

'''
dt1_rbf = pd.read_excel('Summary.xlsx',sheet_name='PortfolioValue')
dt4_rbf = pd.read_excel('Summary.xlsx',sheet_name='AnnualTE')


# average TE
dt4_rbf["annual_TE"].describe()


#worst and best year of IR 
dt4_rbf['Date'][dt4_rbf['annual_IR']==dt4_rbf['annual_IR'].min()]
dt4_rbf['Date'][dt4_rbf['annual_IR']==dt4_rbf['annual_IR'].max()]

'''




'''
#tuning rbf parameter 
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA
import numpy as np

from sklearn.metrics import mean_squared_error
def MSE(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)


param_grid = [{
            "gamma": np.linspace(0.03, 4, 20),
            "kernel": ["rbf","poly" ]  
            }]

kpca=KernelPCA(fit_inverse_transform=True, n_jobs=-1)  # n_jobs using how many parallel processor 
grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=MSE) #cv: how many fold 
clf = grid_search.fit(X)
clf.best_estimator_
clf.best_params_
'''
















