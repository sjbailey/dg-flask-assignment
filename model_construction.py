#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 20:05:58 2021

@author: Samuel Bailey
As part of my submission for Week 4 Assignment of the Data Glacier Virtual Internship
"""

import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

#%%
f = open("old_faithful.csv")
data = np.loadtxt(f, delimiter=",", skiprows = 1, usecols = (1,2)) # load old faithful data into an np.array object
eruptions = np.array([x[0] for x in data])                         # the first column is the desired response variable containing eruption durations
waiting = np.array([x[1] for x in data]).reshape((-1, 1))          # the predictor variable is the time between eruptions
f.close()                                                          

#%%
model = LinearRegression()
model.fit(waiting,eruptions)                                       # the same model was fitted in R and found to be acceptable

#%%
pickle.dump(model, open("model.pkl", "wb"))