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

#仅限LA_1x4
for dtnm in range(7,8):
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
    inter_sec = ['269390046', 'cluster_16298125_4757166089', '2200742494', '361142995']
    control = []
    c_tot = []
    idx= [0, 1,  2,  3,  4,  5,  6,  7,  8 ]

    #0
    c1 = [4, 61, 31]
    c2 = [4, 15, 32, 20, 10]
    c3 = [4, 40, 50, 25, 10]
    c4 = [4, 55, 40, 40, 20]

    #1
    # c1 = [4, 91, 31]
    # c2 = [4, 45, 90, 20, 20]
    # c3 = [4, 45, 90, 20, 20]
    # c4 = [4, 45, 90, 20, 20]
    # # #
    # # # 2
    # c1 = [4, 45, 31]
    # c2 = [4, 45, 90, 20, 20]
    # c3 = [4, 45, 90, 20, 20]
    # c4 = [4, 45, 90, 20, 20]
    # # #
    # # # #3
    # c1 = [4, 61, 31]
    # c2 = [4, 60, 32, 20, 10]
    # c3 = [4, 40, 90, 25, 10]
    # c4 = [4, 55, 90, 20, 20]
    #
    # # 4
    c1 = [4, 45, 31]
    c2 = [4, 45, 90, 25, 25]
    c3 = [4, 45, 90, 25, 25]
    c4 = [4, 45, 90, 25, 25]

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
        y.append(sum/150)
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
for i in range(1,time):
    y[i]=y[i]*1.1

import matplotlib.pyplot as plt
import numpy as np
#设定画布。dpi越大图越清晰，绘图时间越久
fig = plt.figure(figsize=(5,10), dpi=300)
#绘图命令
plt.plot(x, y, lw=3, ls='-', c='b', alpha=0.1)
plt.ylim(0,4000)
plt.xlabel("Time After") #X轴标签
plt.ylabel("Congestion Counts") #Y轴标签
plt.subplots_adjust(bottom=0.15)
plt.legend('Before')
#plt.title("Transportation on 2x2") #标题

plt.plot()
#show出图形
plt.show()
#保存图片
fig.savefig("data2_9")

np.savetxt('x2_9.txt', x, fmt='%d', delimiter=' ')

np.savetxt('y2_9.txt', y, fmt='%d', delimiter=' ')