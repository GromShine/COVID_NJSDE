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




class LSTM_policy(nn.Module):

    def __init__(self, hidden_size=32, seq_len=10, action_dim=10, temperature=5, A_matrix=0):
        
        super(LSTM_policy, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.A = torch.Tensor(A_matrix).view(-1)
        #print("self_h",self.hidden_size)
        
        
        #print((self.A).shape[0])
        
        #k1: consecutive lock down
        #k2: cumulative lock down
        self.k1 = 2 # successive
        self.k2 = 6 # cumulative
        self.k3 = self.action_dim * 0.8 # 80% open within state
        self.k4 = 5 # historical open 5 times, but when to stop
        
        #pre set 
        self.act_count = np.zeros(action_dim)
        self.last_action = np.ones(action_dim) * 2
        self.lock_time = np.zeros(action_dim)
        self.open_county_num = 0
        
        #print(self.last_action)
        #print(self.act_count)

        # Note we should use LSTMCell
        self.lstm_cell = nn.LSTMCell(self.action_dim, self.hidden_size, bias=True)
        # a fully-connect layer that outputs a distribution over the next token, given the RNN output
        self.V = nn.Linear(self.hidden_size, self.action_dim, bias=True)
        # a nonlinear activation function
        self.sigmoid = nn.Sigmoid()
        self.temperature = torch.Tensor([temperature])
        
        #tmp_h = torch.zeros(1, self.hidden_size).float() # batch_size = 1
        #self.last_pi = self.sigmoid(self.temperature * self.V(tmp_h))
        self.last_pi = torch.zeros(action_dim)
        
        self.iter_time = 0
        self.cum_subtraction_pi = 0
        self.lamda = 0.01
        
        

    def compute_dynamic_mask(self, cur_action):
        #print('self_actcnt',self.act_count)
        dynamic_mask=np.zeros(self.action_dim)
        #print(len(dynamic_mask))
        tar = 0
        #print(len(cur_action[0]))
        
        for i in range(len(cur_action[0])):
            if cur_action[0][i] != tar:
                self.act_count[i]=0
                self.open_county_num = self.open_county_num + 1
            else:                
                if self.last_action[i] == cur_action[0][i]:
                    self.act_count[i] = self.act_count[i] + 1
                    if(self.act_count[i] >= self.k1):
                        dynamic_mask[i] = 1 - tar
                        self.act_count[i] = 0
                else:
                    self.act_count[i]=0
                    
            if cur_action[0][i] == tar:
                self.lock_time[i] = self.lock_time[i] + 1
                if self.lock_time[i] > self.k2:
                    dynamic_mask[i] = 1 - tar
                    
        self.last_action = cur_action[0]
        return dynamic_mask
        
    def generate_action(self):
        
        self.act_count = np.zeros(self.action_dim)
        self.last_action = np.ones(self.action_dim) * 2
        self.lock_time = np.zeros(self.action_dim)
        
        action_seq = []
        action_prob_seq = []
        hx = torch.zeros(1, self.hidden_size).float() # batch_size = 1
        cx = torch.zeros(1, self.hidden_size).float()
        #print(self.seq_len)
        
        tar = 0
        for i in range(self.seq_len):
            self.open_county_num = 0
            
            pi = self.sigmoid(self.temperature * self.V(hx)) # nonlinear mapping to a real value in between 0-1
            if self.open_county_num>self.k3:
                for mi in range(self.action_dim):
                    if self.last_action[mi] == tar:
                        pi[mi] = pi[mi] + (1-pi[mi])*0.5
                        
            #print('prob', pi)
            #print('type_pi',type(pi))
            
            action = torch.bernoulli(pi)
            
            
            dynamic_mask = self.compute_dynamic_mask(action)
            
            #print(type(pi))
            
            #print('pi',pi)
            for mi in range(len(dynamic_mask)):
                if dynamic_mask[mi]== 1-tar:
                    action[0][mi] = 1-tar
            
            new_pi = pi.clone()
            
            #print(i,action[0][0])
            
            
            tmp_a=np.zeros(len(dynamic_mask))
                
            cumu_adding_pi = 0
            cumu_no_modify_num = len(dynamic_mask)
            
            for mi2 in range(len(dynamic_mask)):
                if dynamic_mask[mi2]==1 - tar:
                     tmp_a[mi2]=(1-tar-pi[0][mi2])*0.8
                     #cumu_adding_pi = cumu_adding_pi + tmp_a[mi2]
                     #cumu_no_modify_num = cumu_no_modify_num - 1
                    
            '''
            if cumu_no_modify_num != 0:
                avg_modify_pi = cumu_adding_pi / cumu_no_modify_num 
            else:
                avg_modify_pi = 0
            '''        
            for mi3 in range(len(dynamic_mask)):
                if dynamic_mask[mi3]== 1 - tar:
                    new_pi[0][mi3] = pi[0][mi3]+tmp_a[mi3]
                '''else:
                    new_pi[0][mi3] = new_pi[0][mi3] - avg_modify_pi
                    if new_pi[0][mi3]>1:
                        new_pi[0][mi3]=1
                    if new_pi[0][mi3]<0:
                        new_pi[0][mi3]=0
                '''    
            #for mi4 in range(len(dynamic_mask)):
            #    if dynamic_mask[mi4]==1:
            #        new_pi[0][mi4] = 1 
            
            
            
            #print(i,new_pi)
            next_input = action * self.A
            #next_input = action * self.A * torch.Tensor(dynamic_mask)
            
            #next_input = torch.Tensor(dynamic_mask) * self.A
            
            #print('next_input shape[0]',next_input.shape[0])
            #print('next_input shape[1]',next_input.shape[1])
            action_seq.append(action)
            #action_prob_seq.append(pi)
            action_prob_seq.append(new_pi)
            hx, cx = self.lstm_cell(next_input, (hx, cx))
        

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


    def cost(self, ini_intensity,action_prob_seq, cur_decision_time):
        num_decision = len(action_prob_seq)
        cost = []
        pre_intensity = torch.Tensor(ini_intensity)
        for i in range(num_decision):

            A_new = self.A_matrix * action_prob_seq[i].view(self.num_dimension, self.num_dimension)

            # approximate the integration
            integration = torch.zeros(self.num_dimension, self.num_dimension)
            for time_grid in np.arange(cur_decision_time + i*self.decision_interval, cur_decision_time + (i+1)*self.decision_interval, self.grid):
                integration = integration + self.grid * torch.matrix_exp((A_new - self.beta * self.I) * time_grid)

            one_stage_cost = torch.sum(torch.mm(integration, pre_intensity)) + torch.dot(action_prob_seq[i].view(-1), self.cost_Matrix.view(-1))
            pre_intensity = self.mean_field_intensity(pre_intensity, A_new)
            cost.append(one_stage_cost)
        return cost


class trajectory_optimizer():

    def __init__(self, lr=0.001, iter_pg=10000, batch_pg=16):

        
        
        #lreprobility``
        #np.random.seed(1)
        
        ## generate
        # self.A_matrix = 0.1*np.random.random((self.num_dimension, self.num_dimension))
        
        
        #self.A_matrix = torch.load("result4_02.txt")    
        #self.A_matrix = self.A_matrix.detach().numpy()
        
        self.A_matrix = np.loadtxt('result_0.txt',delimiter=',')
        
        print(len(self.A_matrix))
        
        self.num_dimension = len(self.A_matrix)
        #print(self.A_matrix)
        #print((self.A_matrix).shape[0])
        #print((self.A_matrix).shape[1])
        
        
        
        '''
        population=[213505,125927,563301,787038,70529,466647,161361,1605899,703740,518597,801162,826655]
     
        self.Cost_matrix = np.random.random((len(population),len(population)))
        for i in range(len(population)):
            for j in range(len(population)):
                if i!=j:
                    self.Cost_matrix[i][j]=population[i]+population[j]
                else:
                    self.Cost_matrix[i][j]=population[i]
        self.Cost_matrix=self.Cost_matrix/200000
        '''
        #Max_Cost = np.max(self.Cost_matrix)
        #Min_Cost = np.min(self.Cost_matrix)
        #self.Cost_matrix=(10*self.Cost_matrix-Min_Cost)/(Max_Cost-Min_Cost)   
        #self.Cost_matrix = np.random.randint(0, 10, (self.num_dimension, self.num_dimension))
        self.Cost_matrix = np.ones((self.num_dimension, self.num_dimension))
        #print(self.Cost_matrix)
        
        
        
        self.ini_intensity = np.random.random((self.num_dimension, 1))
        
        #print(self.A_matrix)
        #print(self.Cost_matrix)
        #print(self.ini_intensity)
        
        plot1 = plt.figure(1)
        plt.imshow(self.A_matrix, cmap='hot', interpolation='nearest', vmin=0, vmax=5)
        plot2 = plt.figure(2)
        plt.imshow(self.Cost_matrix, cmap='hot', interpolation='nearest',vmin=0, vmax=10)
        plt.show()

        self.cur_decision_time = 1
        self.num_stages = 12

        self.LSTM_policy = LSTM_policy(seq_len=self.num_stages, action_dim=np.square(self.num_dimension),A_matrix=self.A_matrix)
        self.compute_objective = compute_objective(num_dimension=self.num_dimension, beta=1, cost_Matrix=self.Cost_matrix,  decision_interval=1, grid=0.05, A_matrix=self.A_matrix)
        self.lr = lr
        self.iter_pg = iter_pg
        self.batch_pg = batch_pg

        #
        seq_cnt = 0
        optimizer = torch.optim.Adam(self.LSTM_policy.parameters(), lr=self.lr)
        for iter in range(self.iter_pg):
            action_seq, action_prob_seq = self.LSTM_policy.generate_action()
            cost = self.compute_objective.cost(self.ini_intensity, action_prob_seq, self.cur_decision_time)
            loss = torch.sum(torch.stack(cost))
            optimizer.zero_grad()
            if seq_cnt==0:
                tmp_sq = action_prob_seq[0]
                last_sq = action_prob_seq[0]
            else:
                #print(action_prob_seq)
                #print(last_sq)
                tmp_sq = action_prob_seq[0] - last_sq[0]
                last_sq = action_prob_seq[0]
                #print("act_seq",tmp_sq)
            seq_cnt = seq_cnt+1
            
            #torch.autograd.set_detect_anomaly(True)            
            #with torch.autograd.detect_anomaly():
            loss.backward(retain_graph=True)
            
            optimizer.step()
            
            if iter % 10 == 0:
                print('current cost is', loss)
                print("act_seq",tmp_sq)
                print(action_seq[0])
                fig, ax = plt.subplots(4,3,figsize=(9,12))
                for tmp_i in range(4):
                    for tmp_j in range(3):
                        ax[tmp_i][tmp_j].imshow(action_seq[tmp_i*3+tmp_j].detach().numpy().reshape((self.num_dimension, self.num_dimension)), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
                
                plt.pause(0.001)
        plt.show()


#### main ######
torch.autograd.set_detect_anomaly = True
trajectory_optimizer()