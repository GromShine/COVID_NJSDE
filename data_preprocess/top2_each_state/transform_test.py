# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:36:21 2022

@author: Qixiang Wang_UTD
"""


import os
import pickle
import shutil
import numpy as np

def read_event_time2(scale=1.0, h_dt=0.0, t_dt=0.0):
    a_file = open('./ver_num/California_Riverside_06065_2021-06-28_2021-12-28_15.pkl', "rb")
    data = pickle.load(a_file)
    
    
    time_seqs = []
    time_mark = {}
    tmp_time_seq = []
    for i in data.keys():           #遍历所有维度
        for j in data[i]:           #对于每个维度下的每个时间点
            tmp_time_seq.append(j)  #由于时间点已经处理过，不存在相同时间点
            time_mark[j] = i        #记录时间点j的所属维度i
    sort_tmp_time_seq = sorted(tmp_time_seq)    #从小到大排序时间点
    
    time_seqs.append(sort_tmp_time_seq)  
    
    
    
    tmin = min([min(seq) for seq in time_seqs]) #找出最小时间点
    tmax = max([max(seq) for seq in time_seqs]) #找出最大时间点
    
    mark_seqs=[]
    tmp_mark_seqs=[]
    
    for i in sort_tmp_time_seq:
        tmp_mark_seqs.append(time_mark[i])      #向标记序列中依次加入当前时间点的所属维度
    
    mark_seqs.append(tmp_mark_seqs)

    # marks to mark_id
    m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}

    # [[(t1_1,mk1_1),(t2_1,mk2_1)],[(t1_2,mk1_2),(t2_2,mk2_2)],[]]
    # 时间归一化，最小时间从h_dt=1.0开始
    evnt_seqs = [[((h_dt+time-tmin)*scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for time_seq, mark_seq in zip(time_seqs, mark_seqs)]
    
    return evnt_seqs, (0.0, ((tmax+t_dt)-(tmin-h_dt))*scale)

print(read_event_time2())