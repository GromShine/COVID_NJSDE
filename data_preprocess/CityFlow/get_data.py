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

#获取congestion data

#4,6
#for dtnm in range(12,14):
for dtnm in range(8,9):
#for dtnm in range(len(datanames)):
    #name = 'LA_1x4'
    name = datanames[dtnm]
    path = './data/'+name
    print(name)
    file = path+'/'+'replay.txt'

    time = 8000
    time_width = 1
    eng = cityflow.Engine(path + '/config.json', thread_num=8)
    #eng = cityflow.Engine(path + '/config_.json', thread_num=8)
    #让每条lane都no flow，就可以看每条路的红绿灯情况
    light = []
    inter_sec = ['intersection_1_1', 'intersection_1_2', 'intersection_2_1', 'intersection_2_2']
    control = []
    c_tot = []
    idx= [0, 1,  2,  3,  4,  5,  6,  7,  8 ]

    #0，5369
    c1 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c2 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c3 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    c4 = [5, 30, 30, 30, 30, 30, 30, 30, 30]

    # #9，7006
    # c1 = [5, 90, 90, 30, 30, 90, 90, 30, 90]
    # c2 = [5, 90, 90, 30, 30, 90, 90, 30, 90]
    # c3 = [5, 90, 90, 30, 30, 30, 90, 30, 90]
    # c4 = [5, 90, 90, 30, 30, 30, 90, 30, 90]

    c_tot.append(c1)
    c_tot.append(c2)
    c_tot.append(c3)
    c_tot.append(c4)
    for i in range(4):
        for j in range(1,len(c_tot[i])):
            c_tot[i][j] = c_tot[i][j]+c_tot[i][j-1]
    print(c_tot)

    dic_tot = []
    for k in range(4):
        dict_t = {}
        begin=0
        for i in range(len(c_tot[k])):
            for j in range(begin, c_tot[k][i]):
                dict_t[j] = i
            begin = c_tot[k][i]
        dic_tot.append(dict_t)
    print(dic_tot)

    min=-1

    x=[]
    y=[]
    time = 5000
    c_tmp=[]
    out=[]
    cnt=[]
    for i in range(24):
        out.append(c_tmp)
        cnt.append(0)
    #print(len(out),out)
    #exit(0)
    for i in range(time):
        x.append(i)
        cur_light = []
        for j in range(len(inter_sec)):
            light_state = dic_tot[j].get(i%c_tot[j][-1])
            eng.set_tl_phase(inter_sec[j], light_state)
            cur_light.append(light_state)

        light.append(cur_light)

        wait_cnt = eng.get_lane_waiting_vehicle_count()

        idx = 0
        for c in wait_cnt.keys():
            if idx%3!=2:
                cnt[int(idx/3)]+=wait_cnt.get(c)
            if idx%3==1:
                while(cnt[int(idx/3)]>10):
                    cnt[int(idx / 3)] = cnt[int(idx/3)] - 10
                    out[int(idx/3)].append(i)
                cnt[int(idx/3)]=0
            idx = idx + 1

        #print(len(wait_cnt))
        #print(wait_cnt)
        #exit(0)
        sum=0
        for k in wait_cnt.values():
            sum+=int(k)
        y.append(sum)
        if(eng.get_vehicle_count()==0):
            print('end',i)
            if i!=0:
                break
        #print(wait_cnt)
        eng.next_step()
    #print(out)
    np.savetxt('data2.txt', out[0], fmt='%d', delimiter=' ')

