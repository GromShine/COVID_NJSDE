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
        for i in range(num_decision):
            for dtnm in range(2,3):
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

                time = 5000
                for k in range(time):
                    cur_light = []
                    for j in range(len(inter_sec)):
                        if action_seq[i]==1:
                            light_state = dic_tot[j].get(k%c_tot[j][-1])+1
                            eng.set_tl_phase(inter_sec[j], light_state)
                            cur_light.append(light_state)
                        else:
                            light_state = dic_tot[j].get(k%c_tot[j][-1])
                            eng.set_tl_phase(inter_sec[j], light_state)
                            cur_light.append(light_state)

                    light.append(cur_light)

                    wait_cnt = eng.get_lane_waiting_vehicle_count()
                    #print(wait_cnt)
                    sum=0
                    for k in wait_cnt.values():
                        sum+=int(k)
                        cost+=int(k)
                    if eng.get_vehicle_count()>min:
                        min=eng.get_vehicle_count()
                    if(eng.get_vehicle_count()==0):
                        print('end',k)
                        if k!=0:
                            print(min)
                            break
                    #print(wait_cnt)
                    eng.next_step()
        return cost


class trajectory_optimizer():

    def __init__(self, lr=0.001, iter_pg=10000, batch_pg=16):

        self.num_dimension = 15
        
        #lreprobility
        #np.random.seed(1)
        
        ## generate
        self.A_matrix = torch.load("result4_02.txt")
        
        #print(self.A_matrix)
        self.A_matrix = self.A_matrix.detach().numpy()
        #print(self.A_matrix)
        #print((self.A_matrix).shape[0])
        #print((self.A_matrix).shape[1])
        
        
        self.Cost_matrix = np.random.randint(0, 10, (self.num_dimension, self.num_dimension))
        self.ini_intensity = np.random.random((self.num_dimension, 1))
        
        #print(self.A_matrix)
        print(self.Cost_matrix)
        #print(self.ini_intensity)
        
        plot1 = plt.figure(1)
        plt.imshow(self.A_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=5)
        plot2 = plt.figure(2)
        plt.imshow(self.Cost_matrix, cmap='hot', interpolation='nearest',vmin=0, vmax=10)
        plt.show()

        self.cur_decision_time = 1
        self.num_stages = 30

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
            print('current cost is', loss)
            if iter % 100 == 0:
                plt.figure(1)
                plt.imshow(action_prob_seq[0].detach().numpy().reshape((self.num_dimension, self.num_dimension)), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
                plt.pause(0.5)
        plt.show()





#### main ######

trajectory_optimizer()