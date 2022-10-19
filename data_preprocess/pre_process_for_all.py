# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 09:04:26 2022

@author: Qixiang Wang_UTD
"""
# Given the number of a county, start time, end time, and the number of nearby counties
# ouput an covid cases dataset centered on that county


dic = {}
county_idx = {}
county_fips = []
loc_x = []
loc_y = []

tot = 0

# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/
# Geo Point, GeoID
for line in open('county_fips_geo.txt', encoding='gb18030', errors='ignore'):
    # read each county longitude and latitude and unique ID(fip)    
    a = line.split()
    county_idx[a[2]] = tot
    county_fips.append(a[2])
    
    loc_x.append(float(a[1]))
    loc_y.append(float(a[0]))
    
    cord = (float(a[0]),float(a[1]))
    dic[a[2]] = cord
    
    tot += 1
    
fips_state = {}
fips_county = {}
county_state_fips = {}
county_num=0
fips = []

for line in open('county_state_fips.txt'):
    #读取county, 所在state, fip
    #if a[2]=='':
    #    print(a)
    a = line.split('\n')[0].split(',')
    fips_state[a[2]] = a[1]
    fips_county[a[2]] = a[0]
    county_state_fips[a[0]+a[1]]=a[2]
    fips.append(a[2])
    county_num += 1
    

import numpy as np

'''
dis = np.eye(tot,dtype=float)
#dis = [[0]*tot] * tot
#print(type(dis))

import math

#计算任意两个county之间的距离并存储
def cal_dis(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


for i in range(tot):
    for j in range(tot):
        dis[i,j] = cal_dis(loc_x[i], loc_x[j], loc_y[i], loc_y[j])

print(dis)

np.savetxt("county_dis.txt", dis, fmt="%f", delimiter=",")
'''

#直接调用已经存储好的county距离,无需二次计算
dis2 = np.loadtxt("county_dis.txt",delimiter=",")

'''
print(dis2)

if dis.all() == dis2.all():
    print('ok')
'''

def text_save(filename, data):
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i])+'\n'
        file.write(s)
    file.close()
    
import datetime
    
def str_to_dt(s):
    return datetime.datetime.strptime(s,'%Y-%m-%d')

import pickle

def save_dict(data,name):
    a_file = open(name+".pkl", "wb")
    pickle.dump(data, a_file)
    a_file.close()


def query_by_fips(fips, st_date, end_date, neighbor_num):
    # date format '20xx-xx-xx'
    
    #print(fips)
    #print(county_idx[fips])
    
    dis_list = dis2[county_idx[fips]]
    sorted_dis = sorted(enumerate(dis_list), key=lambda x: x[1])
    idx = [i[0] for i in sorted_dis]
    nums = [i[1] for i in sorted_dis]
    #print(county[idx])
    
    tot_len = len(date)
    
    ans = []
    
    dict_q = {}
    
    ans2 = []
    
    dict_data = {}
    dict_e_num = 0
    
    case_split = 50
    
    for j in range(neighbor_num+1):
        tmp_list = []
        for i in range(tot_len):
            if date[i] < end_date and date[i] >= st_date:
                if county_[i]==county_fips[idx[j]]:
                    days = (str_to_dt(date[i])-str_to_dt(st_date)).days
                    #ans.append(str(days)+','+county_[i]+','+str(cases[i]))
    
                    if days == 0:
                        old_cases = cases[i]
                    else:
                        if cases[i] - old_cases >= case_split:
                            tmp_cnt = int ((cases[i] - old_cases)/case_split)
                            for k in range(tmp_cnt):
                                #ans2.append(str(j)+"\t"+str(round(days+k*(1.0/tmp_cnt),2)))
                                tmp_list.append(round(days+k*(1.0/tmp_cnt),2))
                                #print(ans2)
                            old_cases += tmp_cnt * case_split
        #if tmp_list==[]:
        #    print('yes')
        #    exit
        if tmp_list!=[]:
            dict_data[dict_e_num] = tmp_list
            dict_e_num += 1            
    #print(idx[0:neighbor_num])
    # should include itself!
    #print(nums[1:neighbor_num])
    #ans.sort()
    
    return dict_data


def query_by_county_state(county,state, st_date, end_date, neighbor_num):
    # county_id = fips
    query_by_fips(county_state_fips[county+state], st_date, end_date, neighbor_num)
    
date = []
county_ = []
cases = []


for line in open('date_fips_cases.txt', encoding='gb18030', errors='ignore'):   
    a = line.split('\n')[0].split(',')
    date.append(a[0])
    county_.append(a[1])
    cases.append(int(a[2]))
    
      
    
query_county_fips = '78020'
query_county_st = '2021-06-28'
query_county_end = '2021-12-28'
query_county_num = 20
sav_root = 'covid_tot'

import os
if not os.path.exists(sav_root):
    os.mkdir(sav_root)


from tqdm import tqdm

#print(len(fips))
# 3233 counties

for i in tqdm(range(538*0,538*1)):
    
    query_county_fips = fips[i]
    
    #if(fips_state[fips[i]]=='California'):       
    print(i)
    res = query_by_fips(query_county_fips, query_county_st,query_county_end, query_county_num)
    
    #print(len(res))
    if len(res)>2:
        if not os.path.exists(sav_root+'\\'+fips_state[query_county_fips]):
            os.mkdir(sav_root+'\\'+fips_state[query_county_fips])
        
        #print(res)
        #save_doc_name = fips_state[query_county_fips]+"_"+fips_county[query_county_fips]+"_"+query_county_fips+"_"+query_county_st+"_"+query_county_end+"_"+(str)(query_county_num)+".txt"
        save_doc_name = sav_root+'\\'+fips_state[query_county_fips]+'\\'+fips_state[query_county_fips]+"_"+fips_county[query_county_fips]+"_"+query_county_fips+"_"+query_county_st+"_"+query_county_end+"_"+(str)(query_county_num)
        #print(save_doc_name)
        #text_save(save_doc_name,res)
        save_dict(res, save_doc_name)
        #exit
        
#np.savetxt(query_county_name+"_"+query_county_st+"_"+query_county_end+"_"+(str)(query_county_num)+".txt", res, delimiter=",")






