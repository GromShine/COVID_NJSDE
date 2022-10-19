# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:36:21 2022

@author: Qixiang Wang_UTD
"""


import os
import pickle
import shutil

a_file = open("California_San Diego_06073_2021-06-28_2021-12-28_15.pkl", "rb")
data = pickle.load(a_file)

mx = 0
mn = 1000
print(len(data))
for i in data:
    if len(data[i])>mx:
        mx = len(data[i])
    if len(data[i])<mn:
        mn = len(data[i])

print(mx,mn)