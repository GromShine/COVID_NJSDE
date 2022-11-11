# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:36:21 2022

@author: Qixiang Wang_UTD
"""


import os
import pickle
import shutil


#path = 'ver_dis'
path = 'ver_num'

datanames = os.listdir(path)


for i in range(len(datanames)):
    a_file = open(path+'/'+datanames[i], "rb")
    data = pickle.load(a_file)
    print(data)    
    print(len(data))
    exit
    # mx = 0
    # mn = 1000
    # print('len:',len(data))
    # for i in data:
    #     print(len(data[i]))
    #     if len(data[i])>mx:
    #         mx = len(data[i])
    #     if len(data[i])<mn:
    #         mn = len(data[i])
    
    # print('MAX:',mx,'MIN:',mn)