# -*- coding: utf-8 -*-

# extract 2 counties in each state's neighborhood with the least empty data and the most abundant data

import os
import pickle
import shutil

path = 'covid_tot'
datanames = os.listdir(path)

perx = [1/4,1/3,2/5,1/2,3/5,3/4,4/5,1]
tar = 'top2_each_state'

for k in range(len(datanames)):
    lst = []
    lst_len = []
    datanames2 = os.listdir(path+'/'+datanames[k])
    print(datanames2)
    exit
    
    if len(datanames2)==1:
        lst.append(datanames2[0])
        print(lst)
    else:
        for p in range(len(perx)):
            per = perx[p]
            for i in range(len(datanames2)):
                a_file = open(path+datanames[k]+'/'+datanames2[i], "rb")
                data = pickle.load(a_file)
                zero_cnt = 0
                tmp_list_len = 0
                
                #print(data)
                for j in range(len(data)):
                    tmp_list_len += len(data[j])
                    if len(data[j]) == 0:
                        zero_cnt += 1
                #print(tmp_list_len)
                if zero_cnt <= len(data)*per:
                    lst.append(datanames2[i])
                    lst_len.append(tmp_list_len)
                a_file.close()
            if len(lst)>=2:
                #print(per)
                break
        
        sorted_len = sorted(enumerate(lst_len), key=lambda x: x[1], reverse=True)
        idx = [i[0] for i in sorted_len]
        nums = [i[1] for i in sorted_len]    
            
        if len(datanames2)<2:
            for i in range(len(datanames2)):
                print(lst[idx[i]],nums[i])
                shutil.copy(path+datanames[k]+'/'+lst[idx[i]],tar)
        else:
            for i in range(2):
                print(lst[idx[i]],nums[i])
                shutil.copy(path+datanames[k]+'/'+lst[idx[i]],tar)
