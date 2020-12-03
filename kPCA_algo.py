#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:45:33 2020

@author: jameshuang
"""
from sklearn.decomposition import KernelPCA
from sklearn import metrics
import numpy as np
import cvxopt as opt  
from cvxopt import blas, solvers  
import pandas as pd 
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import cool



def rbf_kpca(X, k, gamma):
    sq_dists = pdist(X,metric='sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = np.exp(-gamma*mat_sq_dists)
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K-one_N.dot(K)-K.dot(one_N)+one_N.dot(K).dot(one_N)
    Lambda, Q = np.linalg.eigh(K)
    lam_tol = sum(Lambda)
    alphas = np.column_stack((Q[:, -i]for i in range(1, 1+k)))
    lambdas = [Lambda[-i] for i in range(1, k+1)]
    return alphas, lambdas, lam_tol  #e-vector, e-value, tol varince 

def poly_kpca(X, k, d):
    
    K = metrics.pairwise.polynomial_kernel(X,degree=d)    
    N = X.shape[0]
    one_N = np.ones((N, N))/N
    K = K-one_N.dot(K)-K.dot(one_N)+one_N.dot(K).dot(one_N)
    Lambda, Q = np.linalg.eigh(K)
    lam_tol = sum(Lambda)
    alphas = np.column_stack((Q[:, -i]for i in range(1, 1+k)))
    lambdas = [Lambda[-i] for i in range(1, k+1)]
    
    return alphas, lambdas, lam_tol  #e-vector, e-value, tol varince 


def proj_new_ply(X_new, X, d, alphas, lambdas):
    #x_proj = proj_new_ply(y,X,d,alphas,lambdas)
    
    proj_ret = []    
    k = metrics.pairwise.polynomial_kernel(X,X_new,degree=d)
    
    t = k.shape[1]
    
    for i in range(t):
    
        temp = k[:,i].dot(alphas/np.sqrt(lambdas))
        proj_ret.append(temp)
        
    proj_ret = pd.DataFrame(proj_ret)
    proj_ret.index = X_new.index
    return proj_ret


def proj_new(X_new, X, gamma, alphas, lambdas):
    
    t,n = X_new.shape
    proj_ret = []
    #pd.DataFrame()
    
    for i in range(t):
        x = X_new.iloc[i,:]               
        k = np.exp(-gamma*np.sum((X-x)**2, 1))
        temp = k.dot(alphas/np.sqrt(lambdas))
        proj_ret.append(temp)
    proj_ret = pd.DataFrame(proj_ret)
    proj_ret.index = X_new.index
    return proj_ret


def optimize(returns, mu, wb, beta):  
        
    n = len(returns);  
    returns = np.asmatrix(returns)  
    
    N = 50;
    mus = [10**(5 * t/N - 1.0) for t in range(N)];
    
    #mus = [mu]
    S = opt.matrix(np.cov(returns))   #P

    pbar = opt.matrix(np.mean(returns, axis=1))  
    # Create constraint matrices  
    # -Gx < wb 
    # xGx < tracking 
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix  
    wb = np.matrix(wb).T
    h = opt.matrix(wb)  
    
    A1 = list(np.repeat(1.0, n))
    b1 = 0.0  
    An = beta
    bn = 0.0
    
    A = opt.matrix([A1,An]).T 
    b = opt.matrix([b1,bn])
    
    # Calculate efficient frontier weights using quadratic programming  
    weights = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus] ; 
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
    returns = [blas.dot(pbar, x) for x in weights]  ;
    risks = [np.sqrt(blas.dot(x, S*x)) for x in weights]  
    idx = np.where(np.array(risks)==min(np.array(risks)))[0][0]
    
    wt = np.asarray(weights[idx])
    wt = np.asarray(wt)
    print()
    return wt, weights, returns, risks;

def opt_pt_weight(returns):  
    
    feature,n_obsv = returns.shape
    
    n = len(returns);  
    returns = np.asmatrix(returns)  ;
    N = 50;
    mus = [10**(5 * t/N - 1.0) for t in range(N)];
    #mus = [0.05 + 0.005*t for t in range(N)]
    # Convert to cvxopt matrices  
    S = opt.matrix(np.cov(returns))  ; #P
    
    try:
        S = opt.matrix(np.cov(returns))  ; #P

    except TypeError:
        print("data:",returns.shape)
        print(np.cov(returns))
    
    pbar = opt.matrix(np.mean(returns, axis=1))  ;
    # Create constraint matrices  
    G = -opt.matrix(np.eye(n)) ;  # negative n x n identity matrix  
    h = opt.matrix(0.0, (n ,1))  ;
    A = opt.matrix(1.0, (1, n))  ;
    b = opt.matrix(1.0)  ;

    # Calculate efficient frontier weights using quadratic programming  
    weights = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  
                  for mu in mus] ; 
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER  
    returns = [blas.dot(pbar, x) for x in weights]  ;
    risks = [np.sqrt(blas.dot(x, S*x)) for x in weights]  ;
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
    #m1 = np.polyfit(returns, risks, 2)  ;
    #x1 = np.sqrt(m1[2] / m1[0])  ;
    # CALCULATE THE OPTIMAL PORTFOLIO  
    #wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] ;
    idx = np.where(np.array(risks)==min(np.array(risks)))[0][0]
    wt = np.asarray(weights[idx])
    wt = np.asarray(wt);
    print()
    return wt, weights, returns, risks;

class Algo:
    

    def __init__(self, df, method, p, gma):

        # input pd.Df MONTHLY stock return data, SP500 will be included in final index 
        # x = dt_stock.iloc[i:i+wd,:-1]
        self.df = df.iloc[:,:-1]
        self.y = df.iloc[:,-1]
        self.date = df.index[-1]
        self.table = pd.DataFrame()
        self.explained_var_ratio = []
        self.x_proj = []
        self.alphas = []
        self.lambdas = []
        self.arr = []
        self.method = method
        self.p = p
        self.gma = gma
        
        
        
        
        
    def get_explained_var_ratio(self):
        
        print(self.explained_var_ratio)
        return self.explained_var_ratio
    
    def input_df(self,df):
    
        self.df = df.iloc[:,:-1]
        self.date = df.index[-1]
        #print('data loaded')
    
    def plot_kPCA(self):
        y = self.y
        ret = self.ret
        fig = plt.figure()
        plt.plot(y, label = "SPX")
        plt.plot(ret,label = "kPCA")
        plt.legend()
        plt.show()
        
        
    def predict_kPCA(self,table=False):
        
        y = self.y
        X = self.x_proj
        #X = sm.add_constant(X)
        md = sm.OLS(y, X).fit()
        
        if table==True: 
            print("___Cumulative Explined Variance Ratio___")
            print(self.explained_var_ratio )
            print()
            print(md.summary())
            
        ret = md.predict( X ) 
        #.iloc[-1,:] 
        ret.index = X.index
        self.ret = ret
        
        return ret 

   
    def cal_pf_weight(self,opt_strat):
        
        p = self.p
        gma = self.gma
        
        
        
        
        #demean 
        mu = self.df.mean(axis=0)
        self.df = self.df - mu
        
        
        if (self.method=="poly"):
            #kernalPCA rbf
            self.alphas, lambdas, lam_tol = poly_kpca(self.df, p, gma)
            x_proj =                        proj_new_ply(self.df, self.df, gma, self.alphas, lambdas)
            
            self.x_proj = x_proj
            self.explained_var_ratio = np.cumsum( lambdas/lam_tol )
        
        if (self.method=="rbf"):
            #kernalPCA poly
            self.alphas, lambdas, lam_tol = rbf_kpca(self.df, p, gma)
            x_proj =                        proj_new(self.df, self.df, gma, self.alphas, lambdas)
            
            self.x_proj = x_proj
            self.explained_var_ratio = np.cumsum( lambdas/lam_tol )
        
        
        
        
        M = len(self.df.iloc[0,:])
        W = [0 for e in range(M)]
        
        day = -1 
        # or input array-like strat
        if opt_strat==True:

            wt = opt_pt_weight(self.df)
                
        else:
            
            y = self.y
            X = self.x_proj
            X = sm.add_constant(X)
            md = sm.OLS(y, X).fit()
            
            ret = md.predict( X ) 
            
            pred_all ={}
            for i in range(M):
                tick =  self.df.columns[i]
                stock = self.df.iloc[:,i]
                md = sm.OLS(stock, X).fit()
                pred_all[tick] = md.predict(X)
            
            pred_all = pd.DataFrame(pred_all)
            pred_all["SPX"] = ret
            
            cor = np.corrcoef(pred_all.T)
            
            c = list(np.argsort(cor[-1][:-1]))
            
            
            pred_stock = np.matrix(pred_all)
            pred_stock = pred_stock[:,:-1]
            pred_stock_next = pred_stock[day,:]
            self.pred_stock_next = pred_stock_next
            self.pred_stock = pred_stock
            self.pred_all = pred_all
            ret.index = X.index
            bench = ret[day] 
            
            arr = [] 
            arr2 = []
            for i in range(M):
                if ( pred_stock_next[0,i] > bench ):
                    arr.append(i)
                    c.remove(i)
                else:
                    arr2.append(i)
                        
                if(i==M-1 and len(arr)<2): 
                    arr = arr + c[-3:]
                    
            assert len(arr)>=2
            arr = np.sort(arr)
            data = self.df.iloc[:,arr].T.values
            
            beta =[]
            for i in arr:
                tick =  self.df.columns[i]
                stock = self.df.iloc[:,i]
                Y = sm.add_constant(y)
                md = sm.OLS(stock, Y).fit()
                beta.append(md.params[1])
            
            
            wb = list([1/len(arr) for i in arr])
            beta = list(beta)
            self.beta = beta
            self.arr = arr
            self.wb = wb
            self.data = data 
            
            mu = bench   
            self.mu = mu
            #wt, weights, ret, risks = optimize(data, max(mu,0), wb, beta)
            #wt = wt.squeeze(axis=1)
            
            mu = 2 #risk aversion
            wt = cool.sci_opt(data, beta, mu)
                
            
            aw = wt
            self.aw = aw
            #assert abs(sum(wb + aw) - 1) < 0.0001, "Weight sum !=1:  {}".format(sum(wb + aw)) #check optimization works with constraints
            #assert np.dot(aw, beta) < 0.0001    ##check optimization works with constraints
                
            W = pd.DataFrame(W).T
            W.columns = self.df.columns
            W.iloc[0,arr] = np.squeeze(wb + aw)
            

            
            #print("_____min-var for______")
            #print(list(self.df.columns[arr]))
            #print()
            #print(para_stock)
            #print()
            #print("arr")
            #print(arr)
            

            

        #W = pd.DataFrame(W).T
        #W.columns = self.df.columns
        self.df += mu 
        
        return W

    def update_table(self, W):
        

        self.table = self.table.append(W.T,ignore_index = True)

                
    
    def get_table(self):
        
        '''
                            df.columns (tickers) 
        df.index                weight
        (date)
        
        '''
        
        #print(self.table)
        
        #return self.table
    
    def in_sample_test(self):
        
        weight = self.table.iloc[-1,:]
        ret = np.matmul(self.df.values, weight.values) 
        ret = pd.Series(ret)
        ret.index = self.df.index
        
        return ret 
        
        
        
        
        
        



