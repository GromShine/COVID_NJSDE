# -*- coding: utf-8 -*-
# Given a county ID, start time, end time, and the number of nearby counties
# ouput an covid cases dataset centered on that county
import datetime
import pickle
import numpy as np
import os
from tqdm import tqdm

dic = {}
county_idx = {}     # county idx saved by fips
county_fips = []    # fips saved by idx
loc_x = []          # longitude saved by idx
loc_y = []          # latitude saved by idx

# tot: county num
tot = 0

# https://public.opendatasoft.com/explore/dataset/us-county-boundaries/
# column: Geo Point, GeoID
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
    
fips_state = {}         # state saved by fips
fips_county = {}        # county saved by fips
county_state_fips = {}  # fips saved by str(county+state)
county_num=0
fips = []


for line in open('county_state_fips.txt'):
    # read county's state & fip
    # unknown county does not have fip
    # Kansas City,Joplin, New York City do not have fips
    
    a = line.split('\n')[0].split(',')
    fips_state[a[2]] = a[1]
    fips_county[a[2]] = a[0]
    county_state_fips[a[0]+a[1]]=a[2]
    fips.append(a[2])
    county_num += 1
    



'''
dis = np.eye(tot,dtype=float)
#dis = [[0]*tot] * tot
#print(type(dis))

import math

# calculate each county pair distance
def cal_dis(x1,x2,y1,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


for i in range(tot):
    for j in range(tot):
        dis[i,j] = cal_dis(loc_x[i], loc_x[j], loc_y[i], loc_y[j])

print(dis)

np.savetxt("county_dis.txt", dis, fmt="%f", delimiter=",")
'''

# call the county distance that has been stored, without second calculation
dis2 = np.loadtxt("county_dis.txt",delimiter=",")


def text_save(filename, data):
    file = open(filename,'w')
    for i in range(len(data)):
        s = str(data[i])+'\n'
        file.write(s)
    file.close()
    

    
def str_to_dt(s):
    return datetime.datetime.strptime(s,'%Y-%m-%d')


def save_dict(data,name):
    a_file = open(name+".pkl", "wb")
    pickle.dump(data, a_file)
    a_file.close()


def query_by_fips(fips, st_date, end_date, neighbor_num):
    # use county id(fip) to search
    # date format '20xx-xx-xx'
    
    # get the distance between the target county and all other counties
    dis_list = dis2[county_idx[fips]]
    
    # sort the distance
    sorted_dis = sorted(enumerate(dis_list), key=lambda x: x[1])
    idx = [i[0] for i in sorted_dis]
    nums = [i[1] for i in sorted_dis]
    
    #print(idx[0:10])
    #tmp = idx[0:10]
    #for i in tmp:
    #    print(loc_y[county_idx[county_fips[i]]],',',loc_x[county_idx[county_fips[i]]],',\'',fips_county[county_fips[i]],'\'')
    #print(nums[0:10])
    #exit
    
    tot_len = len(date)
    
    ans = []
    
    dict_q = {}
    
    ans2 = []
    
    # output data dictionary
    dict_data = {}
    # number of counties that at least 1 event happened
    dict_e_num = 0
    
    # teate every N new covid cases as 1 event
    # modify N to control the event sequence length
    case_split = 500
    
    # solve the nearest neighbors' data(include targe county itself)
    for j in range(neighbor_num+1):
        tmp_list = []
        for i in range(tot_len):
            if date[i] < end_date and date[i] >= st_date:
                if county_[i]==county_fips[idx[j]]:
                    #days passed by the last split
                    days = (str_to_dt(date[i])-str_to_dt(st_date)).days
                    
                    if days == 0:
                        old_cases = cases[i]
                    else:
                        if cases[i] - old_cases >= case_split:
                            tmp_cnt = int ((cases[i] - old_cases)/case_split)
                            for k in range(tmp_cnt):
                                tmp_list.append(round(days+k*(1.0/tmp_cnt),2))
                            old_cases += tmp_cnt * case_split
        
        if tmp_list!=[]:
            dict_data[dict_e_num] = tmp_list
            dict_e_num += 1            

    return dict_data


def query_by_county_state(county,state, st_date, end_date, neighbor_num):
    # use county name and state name to search
    # why state name is needed
    # because there r different counties in different states with the same name
    # county_id = fips
    query_by_fips(county_state_fips[county+state], st_date, end_date, neighbor_num)
    
date = []
county_ = []
cases = []

# https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv
# column: data, fips, cases

for line in open('date_fips_cases.txt', encoding='gb18030', errors='ignore'):   
    a = line.split('\n')[0].split(',')
    date.append(a[0])
    county_.append(a[1])
    cases.append(int(a[2]))
    
      
# sample input of query_by_fips(query_county_fips, query_county_st,query_county_end, query_county_num)
query_county_fips = '78020'
query_county_st = '2021-06-28'
query_county_end = '2021-12-28'
tot_time_len = (str_to_dt(query_county_end)-str_to_dt(query_county_st)).days    #总日期跨度

query_county_num = 12

sav_root = 'covid_data_ver_num'

if not os.path.exists(sav_root):
    os.mkdir(sav_root)

# total number of counties with fip: 3233
# the data will be grouped into folders by state

for i in tqdm(range(len(fips))):
    
    query_county_fips = fips[i]
    
    # we can extract 1 state's data to test first
    #if(fips_state[fips[i]]=='California' and fips_county[fips[i]]=='Riverside'):       
    if(fips_state[fips[i]]=='California'):     
        
        rec = {}        #记录当前时间是否已经出现过一次，对于已经出现过的一次事件时间，添加noise
        res = query_by_fips(query_county_fips, query_county_st,query_county_end, query_county_num)
        
        for dim in res.keys():
            for j in range(len(res[dim])):
                res[dim][j] = res[dim][j]/(tot_time_len/100)        
        
        #print(res)
        
        for dim in res.keys():
            for j in range(len(res[dim])):
                if rec.get(res[dim][j])==None:
                    res[dim][j] = round(res[dim][j],3)
                    rec[res[dim][j]] = 1
                    continue;
                else:
                    if j == len(res[dim])-1:
                        while(True):
                            tmp = round(res[dim][j] + 0.1 * np.random.rand(),3)
                            if rec.get(tmp)==None:
                                rec[tmp]=1
                                res[dim][j]=tmp
                                break
                    else:
                        while(True):
                            tmp = round(res[dim][j] + (res[dim][j+1]-res[dim][j]) * np.random.rand(),3)
                            if rec.get(tmp)==None:
                                rec[tmp]=1
                                res[dim][j]=tmp
                                break
        #print(res)
        
        if len(res)>2:
            if not os.path.exists(sav_root+'\\'+fips_state[query_county_fips]):
                os.mkdir(sav_root+'\\'+fips_state[query_county_fips])
            
            save_doc_name = sav_root+'\\'+fips_state[query_county_fips]+'\\'+fips_state[query_county_fips]+"_"+fips_county[query_county_fips]+"_"+query_county_fips+"_"+query_county_st+"_"+query_county_end+"_"+(str)(query_county_num)
            save_dict(res, save_doc_name)
     





