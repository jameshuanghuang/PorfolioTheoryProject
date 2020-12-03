#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:46:37 2020

@author: jameshuang
"""


import numpy as np
import pandas as pd 
import kPCA_algo 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import scipy 
class Backtest:
    

    def __init__(self,wd,dt_stock,data_stock):

        self.wd = wd
        self.dt_stock = dt_stock   #return 
        self.data_stock = data_stock # stock price 
        self.W = pd.DataFrame()
        self.PORTFOLIO  = pd.Series() 
        self.SPX_scaled = pd.Series()

        #initialize algo outsude 
        #ex: go = kPCA_algo.Algo(dt_monthly_return,"poly",p, gma)
        
     
        #temp
        self.m = "poly"
        self.p = 1
        self.gma =1

    def update_para(self,m,p,gma):
            
        self.m = m
        self.p = p
        self.gma =gma
        
    def roll(self,i,opt):
        
        m = self.m 
        p = self.p 
        gma = self.gma 
        
        
        wd = self.wd
        dt_stock = self.dt_stock

        #go = self.Algo
        data = dt_stock.iloc[i:i+wd,:]
        #go.input_df(data)
        go =  kPCA_algo.Algo(data,m,p,gma)    #手動動改這     
        weight = go.cal_pf_weight(opt_strat=opt)
        go.update_table(weight)
    
        return go.table
    
    
    def Roll(self):
        
        end = len(self.dt_stock.index)
        wd = self.wd
        dt_stock = self.dt_stock
        cap = end-wd
        
        W = pd.DataFrame()
        
        for i in range(0,cap):
            self.i = i
            weight = self.roll(i,opt=False).T
            W = W.append( weight, ignore_index = True)
            
        
        W.columns = dt_stock.columns[:-1]
        W.index =   dt_stock.index[wd:wd+len(W.index)]
        
        self.W = W 
        
        
    
    def portfolio_val(self,CASH):
        CASH_c = CASH
        CASH_silly = CASH
        print("Initial Cash:" , CASH)
        W = self.W 
        dt_stock = self.dt_stock
        data_stock = self.data_stock
        
        ret_1 = dt_stock.iloc[:,:-1]+1    
        table = W*ret_1.loc[W.index[0]:W.index[-1]]
        
        #equal weight fund 
        #table_silly = table
        #table_silly.iloc[:,:] = 1/len(dt_stock.columns)
        m, n = table.shape
        e = 1/len(dt_stock.columns)
        table_silly = pd.DataFrame(np.repeat(e,m*n).reshape(m,n))
        table_silly.index = table.index
        table_silly.columns = table.columns
        
        PORTFOLIO = []
        INACTIVE = []
        
        for i in range(len(table.index)):
            temp = np.sum( table.iloc[i]*CASH )
            temp2 = np.sum( table_silly.iloc[i]*CASH_silly )
            PORTFOLIO.append(temp)
            INACTIVE.append(temp2)
            CASH = temp
            CASH_silly = temp2

        PORTFOLIO = pd.Series(PORTFOLIO)
        PORTFOLIO.index = W.index
        
        INACTIVE = pd.Series(INACTIVE)
        INACTIVE.index = W.index

        
        st = PORTFOLIO.index[0]
        ed = PORTFOLIO.index[-1]
    
        val = data_stock.loc[st:ed]
        SPX = val.iloc[:,-1]
        scale = PORTFOLIO[0]/SPX[0]
        SPX_scaled = SPX*scale
        
        self.PORTFOLIO  = PORTFOLIO
        self.SPX_scaled = SPX_scaled
        self.INACTIVE  = INACTIVE
        
        print("End of Value:",PORTFOLIO[-1] )
        print("End of Return:", (PORTFOLIO[-1]-CASH_c)/CASH_c) 
        

    
    
    
    def plot_portfolio_val(self):
        
        PORTFOLIO = self.PORTFOLIO
        SPX_scaled = self.SPX_scaled
        EQLW = self.INACTIVE
        
        fig = plt.figure()
        
        plt.plot(SPX_scaled,color='r',label='SPX')
        plt.plot(PORTFOLIO, color='b',label='Portfolio')
        #plt.plot(EQLW, color='b',label='Eql_Weight')
        plt.title(label="Portfolio_value")
        plt.legend()
        
        fig.show()
        
    def track_error(self):
        

        SPX_scaled = self.SPX_scaled
        PORTFOLIO = self.PORTFOLIO
        
        rP = PORTFOLIO.pct_change()
        rQ = SPX_scaled.pct_change()
        diff =  rP - rQ 
        
        P = PORTFOLIO
        Q = SPX_scaled
        
        r = P.resample('Y').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        b = Q.resample('Y').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        e_y = diff.resample('Y').apply(lambda x: np.std(x)*np.sqrt(252))
        
        IR_y = (r-b)/e_y
        
        plt.subplot(3,1,1)
        plt.plot(diff)
        plt.title('Residual return')
        plt.subplot(3,1,2)
        plt.plot(e_y)
        plt.title('Annual TE')
        plt.subplot(3,1,3)
        plt.plot(IR_y)
        plt.title('Annual IR')
        
        plt.tight_layout()
        plt.legend()
        
        plt.show()
        
    def to_excel(self):

        SPX_scaled = self.SPX_scaled
        PORTFOLIO = self.PORTFOLIO
        INACTIVE = self.INACTIVE
        
        P = PORTFOLIO
        Q = SPX_scaled
        
        rP = PORTFOLIO.pct_change()
        rQ = SPX_scaled.pct_change()
        #Tracking Error: STD(P-B)
        diff =  rP - rQ 
    
        #monthly IR
        r = rP.resample('M').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        b = rQ.resample('M').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        e_m = diff.resample('Y').apply(lambda x: np.std(x)*np.sqrt(252))
        
        IR_m = (r-b)/e_m
        
        #yearly IR
        r = P.resample('Y').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        b = Q.resample('Y').apply(lambda x: (x[-1]/x[0])**(len(x)/252)-1)
        e_y = diff.resample('Y').apply(lambda x: np.std(x)*np.sqrt(252))
        
        IR_y = (r-b)/e_y
        
        T = pd.concat([PORTFOLIO, SPX_scaled, PORTFOLIO - SPX_scaled, INACTIVE],axis=1)
        T.columns=["P","Bench","res","Equal_W_P"]
        
        #daily return 
        T1 = pd.concat([rP, rQ, diff ],axis=1)
        T1.columns=["rP","rBench","r_residual"]
        
        T2 = pd.concat([e_m, IR_m],axis=1)
        T2.columns=["monthlyTE","monthlyIR"]
        
        T3 = pd.concat([e_y, IR_y],axis=1)
        T3.columns=["annual_TE","annual_IR"]
        
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('Summary.xlsx', engine='xlsxwriter')
        
        # Convert the dataframe to an XlsxWriter Excel object.
        T.to_excel(writer, sheet_name='PortfolioValue')
        T1.to_excel(writer, sheet_name='Return')
        T2.to_excel(writer, sheet_name='MonthlyTE')
        T3.to_excel(writer, sheet_name='AnnualTE')
        self.W.to_excel(writer, sheet_name='Weights')
        
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()
        print("Output Data")
        
        
    
    
'''
wd = 12*2
B = Backtest.Backtest(wd,dt_stock,data_stock)    
B.Roll()
B.portfolio_val(10000)
B.plot_portfolio_val()
B.track_error()
B.PORTFOLIO
B.STD
'''




        
        
        
    