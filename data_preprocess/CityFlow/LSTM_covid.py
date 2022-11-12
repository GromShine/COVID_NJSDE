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
        self.k3 = self.action_dim * 0.8 # 80% open within the map
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
        tar = 0                                 #这里设的target是关闭，如果反过来设1，则相当于所有的mask都反过来
                                                #比如连续开放多少天之后要关闭等等
        #print(len(cur_action[0]))
        self.open_county_num = 0 
        for i in range(len(cur_action[0])):     #遍历长度为n*n的action串，这里取[0]是因为action[[act1,act2,...]]
            if cur_action[0][i] != tar:         #如果现在决策是开放
                self.act_count[i]=0             #那么重置连续关闭策略的计数器
                self.open_county_num = self.open_county_num + 1 #更新现在open策略的county数
            else:                                               #如果现在策略是关闭
                if self.last_action[i] == cur_action[0][i]:     #如果现在策略和上次一样是关闭
                    self.act_count[i] = self.act_count[i] + 1   #更新连续关闭策略的计数器
                    if(self.act_count[i] >= self.k1):           #如果连续关闭次数到达界限k1
                        dynamic_mask[i] = 1 - tar               #mask强制设1，下次为开放（与target相反）
                        self.act_count[i] = 0                   #并且重置连续关闭策略的计数器
                else:
                    self.act_count[i]=0                         #这里置0，举例k1=2，当连续第三次关闭的时候
                                                                #此时act_count[i]=2，达到界限，将此时mask置为1
                    
            if cur_action[0][i] == tar:                         #如果现在决策是关闭
                self.lock_time[i] = self.lock_time[i] + 1       #累计封锁时间+1
                if self.lock_time[i] > self.k2:                 #如果封锁时间超过了k2
                    dynamic_mask[i] = 1 - tar                   #mask强制设1，开放
                    
        self.last_action = cur_action[0]                        #更新last_action为当前action
        return dynamic_mask
        
    def generate_action(self):
        
        self.act_count = np.zeros(self.action_dim)          
        self.last_action = np.ones(self.action_dim) * 2     #初始化
        self.lock_time = np.zeros(self.action_dim)          #每个[i,j]通道封锁的时间/次数
        
        action_seq = []
        action_prob_seq = []
        hx = torch.zeros(1, self.hidden_size).float() # batch_size = 1
        cx = torch.zeros(1, self.hidden_size).float()
        #print(self.seq_len)
        
        tar = 0
        #self.open_county_num = 0        #实装该mask的话，需在这里初始化0
        
        for i in range(self.seq_len):       #seq_len, 决策次数
            self.open_county_num = 0        #未实装这个mask,因此在这里设0
            
            pi = self.sigmoid(self.temperature * self.V(hx)) # nonlinear mapping to a real value in between 0-1
            if self.open_county_num>self.k3:                 # 如果地图里80%的county都open了
                for mi in range(self.action_dim):            
                    if self.last_action[mi] == tar:          # 而现在这个county上一次是关闭的
                        pi[mi] = pi[mi] + (1-pi[mi])*0.5     #增加他这次open的概率,但不能增加超过1,这里设的是增加到当前概率和1的中间点
                        
            #print('prob', pi)
            #print('type_pi',type(pi))
            
            action = torch.bernoulli(pi)                    #从n*n长度的概率里sample出n*n长度的0,1决策串
            
            dynamic_mask = self.compute_dynamic_mask(action)#计算mask
            
            #print(type(pi))
            
            #print('pi',pi)
            for mi in range(len(dynamic_mask)):             #遍历mask
                if dynamic_mask[mi]== 1-tar:                #如果mask里是1
                    action[0][mi] = 1-tar                   #action也设1
            
            new_pi = pi.clone()
            
            #print(i,action[0][0])
            
            
            tmp_a = np.zeros(len(dynamic_mask))
                
            cumu_adding_pi = 0
            cumu_no_modify_num = len(dynamic_mask)
            
            for mi2 in range(len(dynamic_mask)):
                if dynamic_mask[mi2] == 1 - tar:                #如果mask里强制设1了
                     tmp_a[mi2] = (1 - tar-pi[0][mi2]) * 0.8    #
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
        
        self.A_matrix = np.loadtxt('result_0.txt',delimiter=',')
        
        print(len(self.A_matrix))
        
        self.num_dimension = len(self.A_matrix)
        #print(self.A_matrix)
        #print((self.A_matrix).shape[0])
        #print((self.A_matrix).shape[1])
        
        '''
        # 人口数据补齐cost_matrix
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
        #归一化,控制cost大小(影响学习结果)
        #Max_Cost = np.max(self.Cost_matrix)
        #Min_Cost = np.min(self.Cost_matrix)
        #self.Cost_matrix=(10*self.Cost_matrix-Min_Cost)/(Max_Cost-Min_Cost)   
        
        #无给定cost_matrix的情况下，随机初始化cost_matrix或全置1
        #self.Cost_matrix = np.random.randint(0, 10, (self.num_dimension, self.num_dimension))
        self.Cost_matrix = np.ones((self.num_dimension, self.num_dimension))
        #print(self.Cost_matrix)
        
        
        
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
        
        self.cur_decision_time = 1      #目前在第几次决策
        self.num_stages = 12            #总决策次数

        self.LSTM_policy = LSTM_policy(seq_len=self.num_stages, action_dim=np.square(self.num_dimension),A_matrix=self.A_matrix)
        self.compute_objective = compute_objective(num_dimension=self.num_dimension, beta=1, cost_Matrix=self.Cost_matrix,  decision_interval=1, grid=0.05, A_matrix=self.A_matrix)
        self.lr = lr
        self.iter_pg = iter_pg
        self.batch_pg = batch_pg

        #seq_cnt = 0
        optimizer = torch.optim.Adam(self.LSTM_policy.parameters(), lr=self.lr)
        for iter in range(self.iter_pg):
            
            #n*n长度的实际决策序列，n*n长度的决策概率
            action_seq, action_prob_seq = self.LSTM_policy.generate_action()
            cost = self.compute_objective.cost(self.ini_intensity, action_prob_seq, self.cur_decision_time)
            loss = torch.sum(torch.stack(cost))
            optimizer.zero_grad()
            '''
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
            '''
            
            #torch.autograd.set_detect_anomaly(True)            
            #with torch.autograd.detect_anomaly():
            loss.backward(retain_graph=True)
            
            optimizer.step()
            
            if iter % 10 == 0:
                print('current cost is', loss)
                #print("act_seq",tmp_sq)
                #print(action_seq[0])
                fig, ax = plt.subplots(4,3,figsize=(9,12))
                for tmp_i in range(4):
                    for tmp_j in range(3):
                        ax[tmp_i][tmp_j].imshow(action_seq[tmp_i*3+tmp_j].detach().numpy().reshape((self.num_dimension, self.num_dimension)), cmap='hot', interpolation='nearest', vmin=0, vmax=2)
                
                plt.pause(0.001)
        plt.show()


#### main ######
torch.autograd.set_detect_anomaly = True
trajectory_optimizer()