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

#仅限syn_2x2_gaussian_500_1h
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
    #c1 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    #c2 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    #c3 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    #c4 = [5, 30, 30, 30, 30, 30, 30, 30, 30]

    # # 1，5048
    # c1 = [5, 60, 60, 30, 30, 60, 30, 60, 30]
    # c2 = [5, 60, 60, 30, 30, 60, 30, 30, 60]
    # c3 = [5, 60, 60, 30, 30, 30, 60, 60, 30]
    # c4 = [5, 60, 60, 30, 30, 30, 60, 30, 60]
    #
    # #2，4897
    # c1 = [5, 80, 80, 30, 30, 80, 30, 80, 30]
    # c2 = [5, 80, 80, 30, 30, 80, 30, 30, 80]
    # c3 = [5, 80, 80, 30, 30, 30, 80, 80, 30]
    # c4 = [5, 80, 80, 30, 30, 30, 80, 30, 80]
    #
    #
    # #3，5453
    # c1 = [5, 45, 45, 45, 45, 45, 45, 45, 45]
    # c2 = [5, 45, 45, 45, 45, 45, 45, 45, 45]
    # c3 = [5, 45, 45, 45, 45, 45, 45, 45, 45]
    # c4 = [5, 45, 45, 45, 45, 45, 45, 45, 45]
    #
    #
    # #4，5551
    # c1 = [5, 90, 90, 30, 30, 90, 30, 90, 30]
    # c2 = [5, 90, 90, 30, 30, 90, 30, 30, 90]
    # c3 = [5, 90, 90, 30, 30, 30, 90, 90, 30]
    # c4 = [5, 90, 90, 30, 30, 30, 90, 30, 90]
    #
    # #5，7521
    # c1 = [5, 60, 30, 30, 30, 60, 30, 30, 60]
    # c2 = [5, 60, 30, 30, 30, 60, 60, 30, 30]
    # c3 = [5, 30, 60, 30, 30, 30, 30, 30, 60]
    # c4 = [5, 30, 30, 30, 30, 30, 30, 30, 30]
    #
    # #6,5041
    # c1 = [5, 70, 70, 30, 30, 70, 30, 70, 30]
    # c2 = [5, 70, 70, 30, 30, 70, 30, 30, 70]
    # c3 = [5, 70, 70, 30, 30, 30, 70, 70, 30]
    # c4 = [5, 70, 70, 30, 30, 30, 70, 30, 70]
    #
    # #7,5148
    # c1 = [5, 50, 50, 30, 30, 50, 30, 50, 30]
    # c2 = [5, 50, 50, 30, 30, 50, 30, 30, 50]
    # c3 = [5, 50, 50, 30, 30, 30, 50, 50, 30]
    # c4 = [5, 50, 50, 30, 30, 30, 50, 30, 50]

    # #8,5966
    # c1 = [5, 60, 90, 30, 30, 60, 90, 30, 60]
    # c2 = [5, 60, 90, 30, 30, 60, 60, 30, 90]
    # c3 = [5, 90, 60, 30, 30, 30, 90, 30, 60]
    # c4 = [5, 90, 90, 30, 30, 30, 90, 30, 90]
    #
    #9，7006
    c1 = [5, 90, 90, 30, 30, 90, 90, 30, 90]
    c2 = [5, 90, 90, 30, 30, 90, 90, 30, 90]
    c3 = [5, 90, 90, 30, 30, 30, 90, 30, 90]
    c4 = [5, 90, 90, 30, 30, 30, 90, 30, 90]

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
    for i in range(time):
        x.append(i)
        cur_light = []
        for j in range(len(inter_sec)):
            light_state = dic_tot[j].get(i%c_tot[j][-1])
            eng.set_tl_phase(inter_sec[j], light_state)
            cur_light.append(light_state)

        light.append(cur_light)

        wait_cnt = eng.get_lane_waiting_vehicle_count()
        #print(wait_cnt)
        sum=0
        for k in wait_cnt.values():
            sum+=int(k)
        y.append(sum/300)
        if eng.get_vehicle_count()>min:
            min=eng.get_vehicle_count()
        if(eng.get_vehicle_count()==0):
            print('end',i)
            if i!=0:
                print(min)
                break
        #print(wait_cnt)
        eng.next_step()

    np.savetxt('light2.txt', light, fmt='%d', delimiter=' ')

for i in range(1,time):
    y[i]=y[i-1]+y[i]
#for i in range(1,time):
#    y[i]=y[i]*0.85

import matplotlib.pyplot as plt
import numpy as np
#设定画布。dpi越大图越清晰，绘图时间越久
fig = plt.figure(figsize=(5,10), dpi=300)
#绘图命令
plt.plot(x, y, lw=3, ls='-', c='b', alpha=0.1)
plt.ylim(0,5000)
plt.xlabel("Time Before") #X轴标签
plt.ylabel("Congestion Counts") #Y轴标签
plt.subplots_adjust(bottom=0.15)
#plt.title("Transportation on 2x2") #标题
plt.legend('Before')

plt.plot()
#show出图形
plt.show()
#保存图片
fig.savefig("data0")

np.savetxt('x0.txt', x, fmt='%d', delimiter=' ')

np.savetxt('y0.txt', y, fmt='%d', delimiter=' ')