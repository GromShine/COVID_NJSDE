import numpy as np
import torch
import torch.nn as nn
import datetime
import random
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import cityflow
from copy import deepcopy
import pickle
import numpy as np

datanames = os.listdir('./data')

name = datanames[2]
path = './data/'+name
print(name)
#exit
file = path+'/'+'replay.txt'

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
    
print(len(lane_belong))
exit
    
    
    
class LSTM_policy(nn.Module):

    def __init__(self, hidden_size=32, seq_len=10, action_dim=10, temperature=1, A_matrix=0):
        super(LSTM_policy, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.A = torch.Tensor(A_matrix).view(-1)
        print((self.A).shape[0])

        # Note we should use LSTMCell
        self.lstm_cell = nn.LSTMCell(self.action_dim, self.hidden_size, bias=True)
        # a fully-connect layer that outputs a distribution over the next token, given the RNN output
        self.V = nn.Linear(self.hidden_size, self.action_dim, bias=True)
        # a nonlinear activation function
        self.sigmoid = nn.Sigmoid()
        self.temperature = torch.Tensor([temperature])


    def generate_action(self):

        action_seq = []
        action_prob_seq = []
        hx = torch.zeros(1, self.hidden_size).float() # batch_size = 1
        cx = torch.zeros(1, self.hidden_size).float()

        for i in range(self.seq_len):
            pi = self.sigmoid(self.temperature * self.V(hx)) # nonlinear mapping to a real value in between 0-1
            # print('prob', pi)
            action = torch.bernoulli(pi)
            next_input = action * self.A
            action_seq.append(action)
            action_prob_seq.append(pi)
            hx, cx = self.lstm_cell(next_input, (hx, cx))
        print('prob', pi)
        return action_seq, action_prob_seq

class compute_objective():

    def __init__(self, num_dimension=10, beta=1, cost_Matrix=0,  decision_interval=1, grid=0.05, A_matrix=0):

        self.num_dimension = num_dimension
        self.beta = beta
        self.I = torch.eye(num_dimension)
        self.decision_interval = decision_interval
        self.grid = grid
        self.cost_Matrix = torch.Tensor(cost_Matrix)
        self.A_matrix = torch.Tensor(A_matrix)

    def mean_field_intensity(self, pre_intensity, A_matrix):
        next_intensity = torch.mm(torch.matrix_exp((A_matrix - self.beta * self.I) * self.decision_interval), pre_intensity)
        return next_intensity


    def cost(self, ini_intensity,action_prob_seq, cur_decision_time,action_seq):
        num_decision = len(action_prob_seq)
        cost = []
        pre_intensity = torch.Tensor(ini_intensity)
        time = 6000
        
        for i in range(num_decision):
            name = datanames[2]     #LA_1x4
            path = './data/'+name
            print(name)
            file = path + '/'+'replay.txt'
            
            time_width = 1
            
            eng = cityflow.Engine(path + '/config_control.json', thread_num=8)
            
            light = []
            
            #所有的十字路口标记符
            inter_sec = ['269390046', 'cluster_16298125_4757166089', '2200742494', '361142995']
            
            #所有的road属于哪个十字路口
            road_itsct={'-185720528':'269390046','468917797.108':'269390046','405366768':'269390046',
                        '-405366771':'cluster_16298125_4757166089','518179569':'cluster_16298125_4757166089','508815228':'cluster_16298125_4757166089','405366764':'cluster_16298125_4757166089',
                        '-405366773':'2200742494','313036847':'2200742494','405367634':'2200742494','508809477#0':'2200742494',
                        '-405367639':'361142995','13400999':'361142995','182456135':'361142995','-506148923#0':'361142995'
                        }
            
            control = []
            c_tot = []
            idx= [0, 1,  2,  3,  4,  5,  6,  7,  8 ]
            
            c1 = [4, 61, 31]            #每个十字路口初始phase数目和时长
            c2 = [4, 15, 32, 20, 10]
            c3 = [4, 40, 50, 25, 10]
            c4 = [4, 55, 40, 40, 20]
            
            c_tot.append(c1)
            c_tot.append(c2)
            c_tot.append(c3)
            c_tot.append(c4)
            
            for i in range(4):                              #遍历每个十字路口i
                for j in range(1,len(c_tot[i])):            #遍历每个十字路口i的每个phase_j
                    c_tot[i][j] = c_tot[i][j]+c_tot[i][j-1] #转换每个phase_j的时间为累计到当前阶段的总时间
            
            print(c_tot)
            
            dic_tot = []
            for k in range(4):                          #遍历每个十字路口k
                dict_t = {}
                begin=0
                for i in range(len(c_tot[k])):          #遍历每个十字路口k的每个phase_i对应的累计时间
                    for j in range(begin, c_tot[k][i]): #遍历每个phase_i开始到结束时间的每个时刻j
                        dict_t[j] = i                   #记录时刻j属于哪个phase_i
                    begin = c_tot[k][i]                 #更新begin为下一个phase的开始时间
                dic_tot.append(dict_t)                  #把每个十字路口的每个时刻属于哪个phase都记录下来
            print(dic_tot)
            
            min=-1
            x=[]
            y=[]
            d_time = int(time/num_decision*i)           #每次模拟都要步进
            
            for ti in range(d_time):   #遍历时间
                cur_light = []
                for j in range(len(inter_sec)):
                    if action_seq[i] == 1:
                        continue
                        
            
            
            
            
                for j in time_list[0].keys():   #遍历红绿灯,j代表当前红绿灯
                    if no_change[j]==0:         #如果当前不是恒绿灯
                        if time_list[ti][j] =='g':   #time_list[i][j]: j车道在i时刻的红绿灯状态
                            cars_on_green_lane = eng.get_lane_vehicles()[j] #获取绿灯车道j上的所有车辆id
                            cars_speed = eng.get_vehicle_speed()            #获取全部车辆speed的字典
                            if ti ==0 or last[j]=='r':
                                tot_time[j].append(ti)                       #记录的是全部绿灯时刻
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
                            
                        elif time_list[ti][j]=='r':
                            if last[j]=='g':
                                tot_end[j][-1] = ti                          #如果上次是绿灯，现在是红灯，更新上次绿灯的结束时间
                                if tot_speed[j][-1]!=0:
                                    tot_avg_speed[j][-1] = tot_speed[j][-1]/tot_cnt[j][-1]
                        last[j] = time_list[ti][j] #更新车道j最新红绿灯状态
                    else:         #如果当前是恒绿灯:比如右转
                        #continue
                        wait = eng.get_lane_waiting_vehicle_count()
                        if wait[j]>0:                                       #恒绿灯出现wait cars就说明是堵车事件，无需再区分
                            if len(tot_time[j])==0 or last[j]==0:           #如果头一回发生恒绿灯有堵车，或者前一时刻没堵车
                                tot_time[j].append(ti)
                                tot_cnt[j].append(wait[j])
                                tot_end[j].append(0)
                            elif last[j]>0:
                                tot_cnt[j][-1] += wait[j]
                        else:
                            if ti!=0 and last[j]>0:
                                tot_end[j][-1] = ti
                        last[j] = wait[j]
                            
                eng.next_step()
            
            
            for i in range(d_time):

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
                y.append(sum)
                if eng.get_vehicle_count()>min:
                    min=eng.get_vehicle_count()
                if(eng.get_vehicle_count()==0):
                    print('end',i)
                    if i!=0:
                        print(min)
                        break
                #print(wait_cnt)

                eng.next_step()
            
            
            A_new = self.A_matrix * action_prob_seq[i].view(self.num_dimension, self.num_dimension)

            # approximate the integration
            integration = torch.zeros(self.num_dimension, self.num_dimension)
            for time_grid in np.arange(cur_decision_time + i*self.decision_interval, cur_decision_time + (i+1)*self.decision_interval, self.grid):
                integration += self.grid * torch.matrix_exp((A_new - self.beta * self.I) * time_grid)

            one_stage_cost = torch.sum(torch.mm(integration, pre_intensity)) + torch.dot(action_prob_seq[i].view(-1), self.cost_Matrix.view(-1))
            pre_intensity = self.mean_field_intensity(pre_intensity, A_new)
            cost.append(one_stage_cost)
        return cost


class trajectory_optimizer():

    def __init__(self, lr=0.001, iter_pg=10000, batch_pg=16):

        
        #np.random.seed(1)
        
        ## generate
        
        self.A_matrix = np.loadtxt('result_0.txt',delimiter=',')
        print(len(self.A_matrix))
        self.num_dimension = len(self.A_matrix)
        
        #print(self.A_matrix)
        #print((self.A_matrix).shape[0])
        #print((self.A_matrix).shape[1])
        
        
        
        #self.Cost_matrix = np.random.randint(0, 10, (self.num_dimension, self.num_dimension))
        self.Cost_matrix = np.ones((self.num_dimension, self.num_dimension))        #假设调度每个十字路口红绿灯的cost均相同
        self.ini_intensity = np.random.random((self.num_dimension, 1))
        
        #print(self.A_matrix)
        #print(self.Cost_matrix)
        #print(self.ini_intensity)
        
        '''
        plot1 = plt.figure(1)
        plt.imshow(self.A_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=5)
        plot2 = plt.figure(2)
        plt.imshow(self.Cost_matrix, cmap='hot', interpolation='nearest',vmin=0, vmax=10)
        plt.show()
        '''
        
        self.cur_decision_time = 1
        self.num_stages = 12

        self.LSTM_policy = LSTM_policy(seq_len=self.num_stages, action_dim=np.square(self.num_dimension),A_matrix=self.A_matrix)
        self.compute_objective = compute_objective(num_dimension=self.num_dimension, beta=1, cost_Matrix=self.Cost_matrix,  decision_interval=1, grid=0.05, A_matrix=self.A_matrix)
        self.lr = lr
        self.iter_pg = iter_pg
        self.batch_pg = batch_pg

        #
        optimizer = torch.optim.Adam(self.LSTM_policy.parameters(), lr=self.lr)
        for iter in range(self.iter_pg):
            action_seq, action_prob_seq = self.LSTM_policy.generate_action()
            cost = self.compute_objective.cost(self.ini_intensity, action_prob_seq, self.cur_decision_time,action_seq)
            loss = torch.sum(torch.stack(cost))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % 10 == 0:
                print('current cost is', loss)
                print(action_seq[0])
                fig, ax = plt.subplots(4,3,figsize=(9,12))
                for tmp_i in range(4):
                    for tmp_j in range(3):
                        ax[tmp_i][tmp_j].imshow(action_seq[tmp_i*3+tmp_j].detach().numpy().reshape((self.num_dimension, self.num_dimension)), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
                
                plt.pause(0.001)
            
        plt.show()





#### main ######
trajectory_optimizer()