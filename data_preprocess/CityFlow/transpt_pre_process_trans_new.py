import os
import cityflow
from copy import deepcopy
import pickle
import numpy as np
import math

def save_dict(data, name):
    a_file = open(name+'.pkl','wb')
    pickle.dump(data,a_file)
    a_file.close()

datanames = os.listdir('./data')
print(datanames)
print(len(datanames))

for dtnm in range(2,3):
#for dtnm in range(len(datanames)):
    #name = 'LA_1x4'
    name = datanames[dtnm]
    path = './data/'+name
    print(name)
    #exit
    file = path+'/'+'replay.txt'
    save_doc_name = path+'/'+name+'_TTS'
    save_txt_name = path+'/'+name+'_TTS.txt'
    

    time = 5500
    time_width = 1
    eng = cityflow.Engine(path + '/config_.json', thread_num=8)
    #eng = cityflow.Engine(path + '/config_.json', thread_num=8)
    #让每条lane都no flow，就可以看每条路的红绿灯情况

    for i in range(time):
        eng.next_step()
        

    lane_belong = {}
    time_list = []                                          #每条lane的红绿灯变化状态轨迹
    for line in open(file):
        whole_street = (line[1:-2]).split(',')
        #开头有一个分号；末尾有一个逗号，
        #print(whole_street)
        tmp_dic = {}                                    #每个时刻存1个字典，保存当下各个road上各个红绿灯的状态信息
        for itsc in whole_street:
            if itsc[-1] == 'i':
                # invisible
                continue

            b = itsc.split()
            #print(b)
            #exit(0)

            # b[0]是road编号，
            for i in range(1,len(b)):
                tmp_dic[b[0]+'_'+str(i-1)] = b[i]       #road编号_第i-1个红绿灯的状态是b[i],(g or r)
                lane_belong[b[0]+'_'+str(i-1)]=b[0]
            
        #print(tmp_dic)
        #exit(0)
        time_list.append(tmp_dic)
    print(len(tmp_dic))
    #for i in range(579,579+61):
    #    print(time_list[i]['-185720528_0'])

    eng.reset()
    eng = cityflow.Engine(path + '/config.json', thread_num=8)
    tot_time={}
    tot_end={}
    tot_speed={}
    tot_cnt={}
    tot_avg_speed={}
    last={}
    for i in time_list[0].keys():   #遍历红绿灯, 已经排除状态为i的灯，不可见灯，表示此处无灯
        tot_time[i]=[]
        tot_end[i]=[]               #记录每次绿灯结束时间，用来制作时间点过程的分布
        tot_speed[i]=[]
        tot_cnt[i]=[]
        tot_avg_speed[i]=[]
        last[i]='i'
    
    no_change = {}                          #存在一些信号灯, 永不变化信号: 右转
    for i in time_list[0].keys():
        no_change[i] = 1                    #找出恒绿灯
    for i in time_list[0].keys():
        for j in range(300):                #5分钟足够判断一个路灯是否恒不变
            if time_list[j+1][i]!=time_list[0][i]:
                no_change[i] = 0
                break
    # for i in time_list[0].keys():
    #     if no_change[i]!=0:               #如果一个lane是恒定绿灯，则no_change[i]==1  
    #         print(i)

    
    #print(time_list[0])
    time = 5500
    for i in range(time):   #遍历时间
        for j in time_list[0].keys():   #遍历红绿灯,j代表当前红绿灯
            if no_change[j]==0:         #如果当前不是恒绿灯
                if time_list[i][j] =='g':   #time_list[i][j]: j车道在i时刻的红绿灯状态
                    cars_on_green_lane = eng.get_lane_vehicles()[j] #获取绿灯车道j上的所有车辆id
                    cars_speed = eng.get_vehicle_speed()            #获取全部车辆speed的字典
                    if i ==0 or last[j]=='r':
                        tot_time[j].append(i)                       #记录的是全部绿灯时刻
                        tot_speed[j].append(0)                      #记录该绿灯时刻下对应的总速度*单位时间
                        tot_cnt[j].append(0)                        #记录该绿灯下总车辆数
                        tot_end[j].append(0)
                        tot_avg_speed[j].append(-1)
                        for k in cars_on_green_lane:                #遍历绿灯车道上的车
                            tot_speed[j][-1] += cars_speed[k]
                        tot_cnt[j][-1] = len(cars_on_green_lane)
                    elif last[j]=='g':
                        for k in cars_on_green_lane:                #遍历绿灯车道上的车
                            tot_speed[j][-1] += cars_speed[k]
                        tot_cnt[j][-1] += len(cars_on_green_lane)
                    
                elif time_list[i][j]=='r':
                    if last[j]=='g':
                        tot_end[j][-1] = i                          #如果上次是绿灯，现在是红灯，更新上次绿灯的结束时间
                        if tot_speed[j][-1]!=0:
                            tot_avg_speed[j][-1] = tot_speed[j][-1]/tot_cnt[j][-1]
                last[j] = time_list[i][j] #更新车道j最新红绿灯状态
            else:         #如果当前是恒绿灯:比如右转
                #continue
                wait = eng.get_lane_waiting_vehicle_count()
                if wait[j]>0:                                       #恒绿灯出现wait cars就说明是堵车事件，无需再区分
                    if len(tot_time[j])==0 or last[j]==0:           #如果头一回发生恒绿灯有堵车，或者前一时刻没堵车
                        tot_time[j].append(i)
                        tot_cnt[j].append(wait[j])
                        tot_end[j].append(0)
                    elif last[j]>0:
                        tot_cnt[j][-1] += wait[j]
                else:
                    if i!=0 and last[j]>0:
                        tot_end[j][-1] = i
                last[j] = wait[j]
                    
        eng.next_step()
    
    # print(len(tot_time))
    # print(len(tot_cnt))
    # print(len(tot_speed))
    # print(len(tot_avg_speed))
    
    
    res = {}            #每条road的事件时间序列的开始时间
    res_end ={}         #每条road上事件时间对应的每次结束时间
    end_time={}         #每条road上最后一次拥塞的结束时间点
    for i in lane_belong.values():  #初始化每条road的事件时间序列为空
        res[i] = {}
        res_end[i] = {}
        
    #print(res)
    
    # for i in tot_time.keys():
    #     #print(no_change[i],i,len(tot_time[i]),tot_time[i])
    #     if no_change[i] == 0:            
    #         for j in range(len(tot_cnt[i])):
    #             print(i,lane_belong[i],tot_time[i][j],tot_avg_speed[i][j],tot_cnt[i][j])
    #     else:
    #         for j in range(len(tot_cnt[i])):
    #             print(i,lane_belong[i],tot_time[i][j],tot_cnt[i][j])
            
    for i in tot_time.keys():                                                   #遍历所有lane_i
        #print(no_change[i],i,len(tot_time[i]),tot_time[i])
        if no_change[i] == 0:                                                   #如果lane_i不是恒绿灯
            for j in range(len(tot_cnt[i])):                                    #遍历lane_i的每次绿灯时车道上的车辆数j
                if tot_cnt[i][j] != 0 and tot_avg_speed[i][j] <1 :              #如果车道上的车数不是0且平均通过速度<1
                    if res[lane_belong[i]].get(tot_time[i][j]) == None:         #res[lane_belong[i]]表示lane_i对应的road的时间事件序列
                                                                                #如果lane_i所属的road没有记录下这个绿灯的开始时间点
                        res[lane_belong[i]][tot_time[i][j]] = tot_cnt[i][j]     #记录下这个绿灯阶段车道在绿灯开始时间点的事件内，车道上的车数*时间
                        res_end[lane_belong[i]][tot_time[i][j]] = tot_end[i][j] #记录下这个绿灯开始时间点对应的结束时间
                    else:
                        res[lane_belong[i]][tot_time[i][j]] += tot_cnt[i][j]    #已经记录过lane_i所属road该时间点的车数*时间的话，直接更新
        else:       #如果是恒绿灯
                    #找出
            for j in range(len(tot_cnt[i])):            #遍历lane_i每次堵车的时候
                if res[lane_belong[i]].get(tot_time[i][j]) == None:
                    res[lane_belong[i]][tot_time[i][j]] = tot_cnt[i][j]
                    res_end[lane_belong[i]][tot_time[i][j]] = tot_end[i][j]
                else:
                    print("here")
                    res[lane_belong[i]][tot_time[i][j]] += tot_cnt[i][j]
            continue
                    
                #print(i,lane_belong[i],tot_time[i][j],tot_avg_speed[i][j],tot_cnt[i][j])
    
    #print(res)
    
    sort_res = {}
    sort_res_end = {}
    for i in res.keys():
        sort_res[i] = dict(sorted(res[i].items(),key=lambda x:x[0]))
        sort_res_end[i] = dict(sorted(res_end[i].items(),key=lambda x:x[0]))
        
    # print(sort_res)
    # print('---------------')
    # print(sort_res_end)
    # print('---------------')
    
    final_res={}
    for i in res.keys():
        final_res[i] = []               #用来记录每条road下的最终congestion时间点过程数据
    
    for i in sort_res.keys():           #遍历每一条road_i
        for j in sort_res[i].keys():    #遍历road_i的每次拥堵时间j
            congest_amount = int(math.ceil(sort_res[i][j]/500))
            stp_len = round( (sort_res_end[i][j] - j)/congest_amount, 3 )
            for k in range(congest_amount):
                final_res[i].append(round((j + k*stp_len)/50,3))    #/50放缩时间跨度
    
    # print(final_res)
    # print('---------------')
    rec = {}
    for dim in final_res.keys():        #adding noise
        for j in range(len(final_res[dim])):
            if rec.get(final_res[dim][j])==None:
                rec[final_res[dim][j]] = 1
                continue;
            else:
                if j == len(final_res[dim])-1:
                    while(True):
                        tmp = round(final_res[dim][j] + 0.1 * np.random.rand(),3)
                        if rec.get(tmp)==None:
                            rec[tmp]=1
                            final_res[dim][j]=tmp
                            break
                else:
                    while(True):
                        tmp = round(final_res[dim][j] + (final_res[dim][j+1]-final_res[dim][j]) * np.random.rand(),3)
                        if rec.get(tmp)==None:
                            rec[tmp]=1
                            final_res[dim][j]=tmp
                            break
             
    pt_sum = 0
    for i in final_res:
       pt_sum += len(final_res[i])
    print(pt_sum)
       
    print(final_res)

    #np.savetxt(save_doc_name,final_res, fmt='%f', delimiter=',')
    save_dict(final_res, save_doc_name)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        