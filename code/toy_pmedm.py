#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:15:05 2020

@author: joe
"""

#%%
import numpy as np
import pandas as pd
from scipy import optimize
#%%

#%%
constraints_ind = pd.read_csv('data/toy_constraints_ind.csv')
constraints_bg = pd.read_csv('data/toy_constraints_bg.csv')
constraints_trt = pd.read_csv('data/toy_constraints_trt.csv')
#%%

#%% build geo lookup
bg_id = constraints_bg.GEOID.astype('str').values
trt_id = [s[0:2] for s in bg_id]
geo_lookup = pd.DataFrame({'bg': bg_id, 'trt': trt_id})
#%% 

#%% PUMS Serials
serial = constraints_ind.SERIAL.astype('str').values
#%%

#%% sample weights
wt = constraints_ind.PERWT.values
#%%

#%% population and sample sizes
N = sum(constraints_bg.POP)
n = constraints_ind.shape[0]
#%%

#%% Individual (PUMS) constraints
excl = ["SERIAL", "PERWT"]
pX = constraints_ind.loc[:, ~constraints_ind.columns.isin(excl)].values
#%%

#%% collect names of est/SE columns
se_cols = constraints_bg.columns[np.where([k.endswith('s') for k in constraints_bg.columns])]
est_cols = constraints_bg.columns[np.where([(k not in se_cols) & (k != 'GEOID') for k in constraints_bg.columns])]
#%%

#%% Geographic constraints
Y1 = constraints_trt.loc[:, est_cols].values
Y2 = constraints_bg.loc[:, est_cols].values
#%%

#%% Error variances 
V1 = np.square(constraints_trt.loc[:, se_cols].astype('float').values)
V2 = np.square(constraints_bg.loc[:, se_cols].astype('float').values)
#%%

#%% Geographic crosswalk
A1 = np.array([1 * (geo_lookup.trt == G).values for G in pd.unique(geo_lookup.trt)])
#%%

#%% Target unit identity matrix
A2 = np.identity(geo_lookup.shape[0])
#%%

#%% Solution space (X-matrix)
X1 = np.kron(np.transpose(pX), A1)
X2 = np.kron(np.transpose(pX), A2)
X = np.transpose(np.vstack((X1, X2)))
#%%

#%% Design weights
q = np.repeat(wt, A1.shape[1], axis = 0)
q = q / np.sum(q)
#%%

#%% Vectorize geo. constraints (Y) and normalize
Y_vec = np.concatenate([Y1.flatten('A'), Y2.flatten('A')]) / N
#%%

#%% Vectorize error variances and normalize
V_vec = np.concatenate([V1.flatten('A'), V2.flatten('A')]) * (n / N**2)
#%%

#%% (Initial coefficients/lambdas)
lam = np.zeros((len(Y_vec),))
#%%

#%% compute allocation (SCRATCH)
# a0 = np.exp(np.matmul(-X, lam))
# a = q * a0
# b = np.dot(q, a0)
# np.divide(a, b)
#%%


#%% compute allocation (function)
def compute_allocation(q, X, lam):
    
    a0 = np.exp(np.matmul(-X, lam))
    
    a = q * a0
    
    b = np.dot(q, a0)
    
    ab = np.divide(a, b)
    
    return ab
#%%

#%% test it
p_hat = compute_allocation(q, X, lam)
p_hat = np.reshape(p_hat, (pX.shape[0], A2.shape[0]))
#%%

#%% compute the target (block group) constraint estimates
Yhat2 = np.matmul(np.transpose(N * p_hat), pX)
#%%

#%% compute the upper (tract) constraint estimates
p_hat_up = np.matmul((p_hat * N), np.transpose(A1))
Yhat1 = np.matmul(np.transpose(p_hat_up), pX)
#%%

#%% Vectorize constraint estimates
Yhat = np.concatenate([Yhat1.flatten('F'), Yhat2.flatten('F')])
#%%

#%% Assemble results
Yres = pd.DataFrame({'Y': Y_vec * N, 'Yhat': Yhat,\
                     'V': np.multiply(V_vec, float(N**2 / n))})
#%%
    
#%% Primal function (SCRATCH)
# w = Yres.iloc[0].Y
# d = Yres.iloc[0].Yhat
# v = Yres.iloc[0].V

# e = d - w

# penalty = e**2 / (2 * v)

# ent = ((n / N) * (w / d) * np.log((w/d)))

# pe = (-1 * ent) - penalty

# print(pe)
#%%

#%% Primal function
def penalized_entropy(w, d, v, n, N):
    
    e = d - w

    penalty = e**2 / (2 * v)

    ent = ((n / N) * (w / d) * np.log((w/d)))

    pe = (-1 * ent) - penalty
    
    return pe

#%%

#%% test it
pe = []

for i in range(Yres.shape[0]):
    
    pei = penalized_entropy(Yres.iloc[i].Y, Yres.iloc[i].Yhat, Yres.iloc[i].V,\
                            n, N)
    pe.append(pei)

pe = np.array(pe)

print(-1 * np.mean(pe))
#%%

#%% Objective function
def neg_pe(lam):
    
    p_hat = compute_allocation(q, X, lam)
    p_hat = np.reshape(p_hat, (pX.shape[0], A2.shape[0]))
    Yhat2 = np.matmul(np.transpose(N * p_hat), pX)
    
    p_hat_up = np.matmul((p_hat * N), np.transpose(A1))
    Yhat1 = np.matmul(np.transpose(p_hat_up), pX)
    
    Yhat = np.concatenate([Yhat1.flatten('F'), Yhat2.flatten('F')])
    
    Yres = pd.DataFrame({'Y': Y_vec * N, 'Yhat': Yhat,\
                     'V': np.multiply(V_vec, float(N**2 / n))})
    
    pe = []

    for i in range(Yres.shape[0]):
    
        pei = penalized_entropy(Yres.iloc[i].Y, Yres.iloc[i].Yhat, Yres.iloc[i].V, n, N)
        pe.append(pei)

    pe = -1 * np.mean(np.array(pe))
    
    return pe
#%%
    
#%% test it
%time neg_pe(lam)
#%%

#%% optimization
%time res = optimize.minimize(neg_pe, x0 = lam, method = 'BFGS', options = {'maxiter': 200})
#%%

#%% blah
# res.x
# res.hess_inv # nice!
#%%

#%% inspect results
lamf = res.x # final lambda 

p_hat = compute_allocation(q, X, lamf)
p_hat = np.reshape(p_hat, (pX.shape[0], A2.shape[0]))
Yhat2 = np.matmul(np.transpose(N * p_hat), pX)

p_hat_up = np.matmul((p_hat * N), np.transpose(A1))
Yhat1 = np.matmul(np.transpose(p_hat_up), pX)

Yhat = np.concatenate([Yhat1.flatten('F'), Yhat2.flatten('F')])

Yres = pd.DataFrame({'Y': Y_vec * N, 'Yhat': Yhat,\
                 'V': np.multiply(V_vec, float(N**2 / n))})
    
Yres['Err'] = Yres.Y - Yres.Yhat    
Yres['MOE_lower'] = Yres.Y - (np.sqrt(Yres.V) * 1.645)
Yres['MOE_upper'] = Yres.Y + (np.sqrt(Yres.V) * 1.645)

# proportion of contstraints falling within 90% Margins of Error
win_moe = (Yres.Yhat >= Yres.MOE_lower) & (Yres.Yhat <= Yres.MOE_upper)
print(sum(win_moe) / Yres.shape[0])
#%%