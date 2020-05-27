#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:15:05 2020

@author: joe
"""

#%%
import os
import numpy as np
import pandas as pd
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
Y_vec = (np.vstack((Y1, Y2)) / N).flatten('A')
#%%

#%% Vectorize error variances and normalize
V_vec = (np.vstack((V1, V2)) * (n / N**2)).flatten('A')
#%%


