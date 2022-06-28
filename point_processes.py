import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules import RunningAverageMeter, ODEJumpFunc
from utils import forward_pass, visualize, create_outpath, read_timeseries
import matplotlib.pyplot as plt

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('point_processes')
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--jump_type', type=str, default='read')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--nsave', type=int, default=50)
parser.add_argument('--dataset', type=str, default='t1')
# for now, big data use 'c1', small data use 't1'

parser.add_argument('--suffix', type=str, default='')

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

#outpath = create_outpath(args.dataset)

if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)
    relu = nn.ReLU()
    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    dim_c, dim_h, dim_N, dt, tspan = 3, 2, 1, 0.05, (0.0, 90.0)
    # dim_c, dim_h, dim_N, dt, tspan = 10, 10, 1, 0.05, (0.0, 90.0)
    # 增加神经网络参数测试
    
    # dim_c,dim_h: In order to better simulate the time series, the latent state z(t)∈ R^n is further split into 
    # two vectors: c(t)∈ R^n1 encodes the internal state, and h(t)∈ R^n2 encodes the memory of 
    # events up to time t, where n = n1 + n2.
    # dim_N,有多少种事件种类,COVID这里暂时只有cases增长一种
    # dt, 时间间隔
    # tspn, 时间范围
    path = "./data/"
    TRAIN = read_timeseries(path + args.dataset + ".csv")
    county_num = len(TRAIN)
    #数据维度
    
    #初始化A-matrix
    A_matrix = Variable(0.01*torch.ones((county_num, county_num)), requires_grad= True)
    
    
    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=20, num_hidden=1, ortho=True, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
    # func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=30, num_hidden=5, ortho=True, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
    # 提高隐藏层、隐藏层维度测试
    
    c0 = torch.randn(dim_c, requires_grad=True)
    h0 = torch.zeros(dim_h)
    it0 = 0
    
    #对微分方程的参数和A_matrix进行优化
    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params':A_matrix},
                            {'params': c0, 'lr': 1.0e-2},
                            ], lr=1e-3, weight_decay=1e-5)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss_meter = RunningAverageMeter()
    # 设置计算移动平均值的函数
    
    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            # clear out gradients for variables
            optimizer.zero_grad()
            # sample a mini-batch, create a grid based on that
            #batch_id = np.random.choice(len(TRAIN), args.batch_size, replace=False)
            batch_id = np.arange(0,county_num,1)
            batch = [TRAIN[seqid] for seqid in batch_id]
            
            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align, A_matrix)
            loss_meter.update(loss.item() / len(batch))
            #计算loss的移动平均值
            
            # backward prop
            func.backtrace.clear()
            loss.backward()
            
            
            for Ai in range(county_num):
                for Aj in range(county_num):
                    with torch.no_grad():
                        if A_matrix[Ai,Aj]<=0:
                            A_matrix[Ai,Aj]=relu(A_matrix[Ai,Aj])+0.0001
            # 可以只在最后一步做此输出
            
            if torch.norm(A_matrix.grad)<1e-4:
                #终止条件
                torch.save(A_matrix,"result.txt")
                break
            print(A_matrix)
            #np.savetxt(r'result.txt',A_matrix.detach().numpy(), fmt='%f', delimiter=',')
            
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item()/len(batch), loss_meter.avg, mete), flush=True)
            if it % 1 == 0:
                plt.figure(1)
                plt.imshow(A_matrix.detach().numpy(), cmap='hot', interpolation='nearest', vmin=0, vmax=0.05)
                plt.pause(0.5)
            
            # step
            optimizer.step()

            it = it+1
            