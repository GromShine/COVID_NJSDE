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

#4,6
#for dtnm in range(12,14):



for dtnm in range(8,9):
#for dtnm in range(len(datanames)):
    #name = 'LA_1x4'
    name = datanames[dtnm]
    path = './data/'+name
    print(name)
    file = path+'/'+'replay.txt'

    time_width = 1
    eng = cityflow.Engine(path + '/config.json', thread_num=8)
    #让每条lane都no flow，就可以看每条路的红绿灯情况

    #所有时刻信号灯状态集
    light = np.loadtxt('light.txt')

    inter_sec=['intersection_1_1','intersection_2_1','intersection_1_2','intersection_2_2']

    for i in range(len(light)):
        cur_light = light[i]
        if i%245<5:
            for j in range(len(inter_sec)):
                eng.set_tl_phase(cur_light[j], 0)
                cur_light.append(0)
        else:
            for j in range(len(inter_sec)):
                eng.set_tl_phase(j,(int)((i-5)/30)+1)
                cur_light.append((int)((i-5)/30)+1)
        light.append(cur_light)
        eng.next_step()
    exit(0)
