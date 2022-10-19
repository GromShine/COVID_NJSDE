# -*- coding: utf-8 -*-

# extract 2 counties in each state's neighborhood with the least empty data and the most abundant data

import os
import pickle
import shutil

path = 'covid_data_ver_num'
datanames = os.listdir(path)

# the less percentage of zero event sequence, the better data quality
perx = [1/10,1/5,1/4,1/3,2/5,1/2,3/5,3/4,4/5,1]
#tar = 'top2_each_state/ver_dis/'
tar = 'top2_each_state/ver_num/'
if not os.path.exists(tar):
    os.mkdir(tar)

# for each state
for k in range(len(datanames)):
    lst = []
    lst_len = []
    datanames2 = os.listdir(path+'/'+datanames[k])
    
    if len(datanames2) == 1:
        lst.append(datanames2[0])
        print(lst)
    else:
        
        for p in range(len(perx)):
            per = perx[p]
            for i in range(len(datanames2)):
                a_file = open(path +'/' + datanames[k] + '/' + datanames2[i], "rb")
                data = pickle.load(a_file)
                zero_cnt = 0
                tmp_list_len = 0
                
                # count zero event sequence number
                for j in range(len(data)):
                    tmp_list_len += len(data[j])
                    if len(data[j]) == 0:
                        zero_cnt += 1
                
                if zero_cnt <= len(data) * per:
                    lst.append(datanames2[i])
                    lst_len.append(tmp_list_len)
                a_file.close()
            
            # if current percentage is enough to get top2 county data
            if len(lst) >= 2:
                print(per)
                break
        
    sorted_len = sorted(enumerate(lst_len), key=lambda x: x[1], reverse=True)
    idx = [i[0] for i in sorted_len]
    nums = [i[1] for i in sorted_len]    
        
    if len(datanames2)<2:
        for i in range(len(datanames2)):
            print(lst[idx[i]],nums[i])
            shutil.copy(path+'/'+datanames[k]+'/'+lst[idx[i]],tar)
    else:
        for i in range(2):
            print(lst[idx[i]],nums[i])
            shutil.copy(path+'/'+datanames[k]+'/'+lst[idx[i]],tar)
