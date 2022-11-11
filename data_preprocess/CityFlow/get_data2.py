import os
import cityflow
from copy import deepcopy
import pickle
import numpy as np
def save_dict(data, name):
    a_file = open(name+'.pkl','wb')
    pickle.dump(data,a_file)
    a_file.close()

datanames = os.listdir('./data')
print(datanames)

#只负责提取congestion数据

for dtnm in range(8,9):
#for dtnm in range(len(datanames)):
    #name = 'LA_1x4'
    name = datanames[dtnm]
    path = './data/'+name
    print(name)
    file = path+'/'+'replay.txt'

    time = 5370
    time_width = 1
    eng = cityflow.Engine(path + '/config_.json', thread_num=8)
    #eng = cityflow.Engine(path + '/config_.json', thread_num=8)
    #让每条lane都no flow，就可以看每条路的红绿灯情况

    inter_sec = ['intersection_1_1', 'intersection_2_1', 'intersection_1_2', 'intersection_2_2']

    control = []
    c_tot = []

    # 全部十字路口的信号灯phase时间
    #0，5369
    c1 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c2 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c3 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c4 = [5, 30, 30, 30, 30, 30, 30, 30, 30]

    c_tot.append(c1)
    c_tot.append(c2)
    c_tot.append(c3)
    c_tot.append(c4)
    #求出各个路口的循环phase累计时间
    for i in range(4):
        for j in range(1,len(c_tot[i])):
            c_tot[i][j] = c_tot[i][j]+c_tot[i][j-1]
    #print(c_tot)

    #确定每个红绿灯每个时间下的红绿灯phase
    dic_tot = []
    for k in range(len(inter_sec)):
        #对每个十字路口
        dict_t = {}
        begin=0
        #设置phase状态i
        for i in range(len(c_tot[k])):
            #对该十字路口相应phase对应的时间段的每一个时刻
            for j in range(begin, c_tot[k][i]):
                #字典形式存储，对应时刻的phase状态
                dict_t[j] = i
            #更新下一个phase起点时间
            begin = c_tot[k][i]
        dic_tot.append(dict_t)
    #print(dic_tot)

    #每个时刻下每个十字路口的phase状态
    light = []
    for i in range(time):
        cur_light = []
        for j in range(len(inter_sec)):
            #更新第j个路口的phase状态，注意循环phase的处理，取模最后一位，即总phase长度
            light_state = dic_tot[j].get(i % c_tot[j][-1])
            eng.set_tl_phase(inter_sec[j], light_state)
            cur_light.append(light_state)

        light.append(cur_light)

        # print(wait_cnt)
        eng.next_step()

    np.savetxt('light.txt', light, fmt='%d', delimiter=' ')


    time_list = []                                          #每条lane的红绿灯变化状态轨迹
    for line in open(file):
        whole_street = (line[1:-2]).split(',')
        #开头有一个分号；末尾有一个逗号，
        #print(whole_street)
        tmp_dic = {}
        for itsc in whole_street:
            if itsc[-1] == 'i':
                # no change anymore
                continue

            b = itsc.split()
            #print(b)
            #exit(0)

            # b[0]是road编号，
            for i in range(1,len(b)):
                tmp_dic[b[0]+'_'+str(i-1)] = b[i]
        #print(tmp_dic)
        #exit(0)
        time_list.append(tmp_dic)

    #for i in range(579,579+61):
    #    print(time_list[i]['-185720528_0'])

    eng.reset()
    eng = cityflow.Engine(path + '/config.json', thread_num=8)

    last_lane_v = {}
    last_light = time_list[0]
    last_g_st_time = {}
    last_r_st_time = {}
    TPP = {}                                #每条lane绿灯阶段的通过车数
    G_st_v_cnt = {}                         #每条lane绿灯开始时的车辆数，用于判断0绿灯通过数时的
    data = {}                               #用来存储最后的congestion TPP
    no_change = {}                          #存在一些信号灯，永不变化信号
    g_dur_time = {}                           #每个绿灯的持续时间
    r_dur_time = {}                           #每个红灯的持续时间
    final_g_dur_time = {}                     #congestion的持续时间

    for i in time_list[0].keys():
        no_change[i] = 1                    #找出长明灯
        last_lane_v[i] = []                 #每条lane的最新状态初始化为空
        TPP[i] = {}                         #每条lane的时间点过程初始化为空
        G_st_v_cnt[i] = {}
        g_dur_time[i] = {}
        r_dur_time[i] = {}

        final_g_dur_time[i] = {}
        data[i] = []
        if time_list[0][i] == 'g':          #检查每条lane在time=0时是否为绿灯，如果是，则设最新事件时间为0
            last_g_st_time[i] = 0
            TPP[i][0] = 0
            dic = eng.get_lane_vehicle_count()
            G_st_v_cnt[i][0] = dic[i]
            g_dur_time[i][0] = 0
        elif time_list[0][i] == 'r':          #检查每条lane在time=0时是否为红灯，如果是，则设最新事件时间为0
            last_r_st_time[i] = 0
            r_dur_time[i][0] = 0

    for i in time_list[0].keys():
        for j in range(time-1):
            if time_list[j+1][i]!=time_list[0][i]:
                no_change[i] = 0
                break
        if no_change[i] == 1:
            no_change[i] = time_list[0][i]

    #print(no_change)

    #print(last_g_st_time)

    for i in range(time-1):
        for j in range(len(inter_sec)):
            light_state = dic_tot[j].get(i % c_tot[j][-1])
            eng.set_tl_phase(inter_sec[j], light_state)
            cur_light.append(light_state)
        eng.next_step()
        dic = eng.get_lane_vehicles()                        #全部lane的车
        dic_cnt = eng.get_lane_vehicle_count()               #全部lane上车的数量
        dict_wait = eng.get_lane_waiting_vehicle_count()     #全部lane上waiting count，针对恒绿灯
        cnt = 0
        for j in dic.keys():                        #j表示第j条lane
            if time_list[0].get(j):                 #如果第j条lane是带红绿灯的lane
                if no_change[j]!='g':               #且非恒绿灯
                    if last_light[j] == 'r' and time_list[i+1][j] == 'r':
                        #print(i,j,r_dur_time[j])
                        r_dur_time[j][last_r_st_time[j]] += time_width
                    elif last_light[j] == 'r' and time_list[i+1][j] == 'g':
                        last_g_st_time[j] = i+1
                        g_dur_time[j][i+1] = 0
                        TPP[j][i+1] = 0             #从红灯转绿灯，初始化当前绿灯lane通过车辆数据为0
                        G_st_v_cnt[j][i+1] = dic_cnt[j]
                        #last_lane_v[j] = dic[j]    #更新lane的最新状态
                        last_light[j] = 'g'
                        r_dur_time[j][last_r_st_time[j]] += time_width
                    elif last_light[j] == 'g' and time_list[i+1][j] == 'g':

                        for k in last_lane_v[j]:
                            if k not in dic[j]:
                                #print('j',j,'TPP[j]',TPP[j])
                                #print(TPP[j][last_g_st_time[j]])
                                TPP[j][last_g_st_time[j]] += 1
                        g_dur_time[j][last_g_st_time[j]] += time_width
                    elif last_light[j] == 'g' and time_list[i+1][j] == 'r':  #g -> r
                        last_r_st_time[j] = i + 1
                        r_dur_time[j][i+1] = 0
                        for k in last_lane_v[j]:
                            if k not in dic[j]:
                                TPP[j][last_g_st_time[j]] += 1
                        last_light[j] = 'r'
                        #print(i,j,g_dur_time[j])
                        g_dur_time[j][last_g_st_time[j]] += time_width
                    last_lane_v[j] = dic[j]         # 更新lane的最新状态
                else:                               # 特殊路段，恒绿灯
                    if dict_wait[j] > 0:
                        data[j].append(i+1)

    #print(TPP)
    #print(G_st_v_cnt)

    for ℹ in TPP.keys():                                #i, 表示第i条lane
        for j in TPP[i].keys():                         #j，表示第i条lane红灯转绿灯开启的时刻
            if TPP[i][j] == 0 and G_st_v_cnt[i][j]!=0:  #TPP[i][j]，表示第i条lane在j时刻开启绿灯之后通过的车数，如果为0，且绿灯开始时车数不为0
                data[i].append(j)
                final_g_dur_time[i][j]=g_dur_time[i][j]

    final_data = deepcopy(data)
    congestion_dur_time = deepcopy(final_g_dur_time)
    par = {}                                              #每个合并的连续区间的父区间
    for i in data.keys():
        if len(data[i])==0:
            final_data.pop(i)
            congestion_dur_time.pop(i)
        else:
            if no_change[i] != 'g':
                par[i] = {}
                for j in range(len(data[i])-1):
                    g_st = data[i][j]                             #start time
                    #print(i,j)
                    g_end = g_st + final_g_dur_time[i][g_st]
                    if par[i].get(g_st) == None:
                        par[i][g_st] = g_st
                    if g_end + r_dur_time[i][g_end] == data[i][j+1]:
                        final_data[i].remove(data[i][j+1])
                        #print(i,j,par[i])
                        par[i][data[i][j+1]] = par[i][g_st]            #将被删除的时间区间父节点设到前一个时间区间
                        #print('|||',i,j)
                        #print(congestion_dur_time[i][g_st])
                        congestion_dur_time[i][par[i][g_st]] += r_dur_time[i][g_end] + final_g_dur_time[i][data[i][j+1]]
                        congestion_dur_time[i].pop(data[i][j+1])
            else:
                final_data.pop(i)
                if data[i]!=[]:
                    final_data[i] = [data[i][0]]
                    last_st_time = data[i][0]
                    last_idx = 0
                    for j in range(1, len(data[i])):
                        if data[i][j] - last_st_time == j - last_idx:
                            congestion_dur_time[i][last_st_time] = j - last_idx
                        else:
                            last_idx = j
                            last_st_time = data[i][j]
                            final_data[i].append(data[i][j])

    #print(len(final_data))
    #print(len(congestion_dur_time))

    #print(final_data)
    print(name,congestion_dur_time)

    sav_loc = './result' +'/'+name+'_result2'

    save_dict(congestion_dur_time,sav_loc)

    #a_file = open(sav_loc+'.pkl','rb')
    #data = pickle.load(a_file)
    #print(data)
    #a_file.close()

    #print(data)
    #print(final_g_dur_time)
    #print(r_dur_time)
