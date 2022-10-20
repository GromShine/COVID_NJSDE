import sys
import subprocess
import signal
import argparse
import random
import numpy as np
import matplotlib
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules import RunningAverageMeter, ODEJumpFunc
from utils import forward_pass, visualize, read_timeseries
import matplotlib.pyplot as plt

signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('stack_overflow')
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--jump_type', type=str, default='read')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--nsave', type=int, default=10)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--dataset', type=str, default='m1')
# small example data use 'm1' : Massachusetts from 20200928 to 20201228
# big example data use 'c1': california from 20200928 to 20201228

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()



def read_event_time(scale=1.0, h_dt=0.0, t_dt=0.0):
    time_seqs = []
    with open('./data/'+args.dataset+'_time.txt') as ftime:
        seqs = ftime.readlines()
        for seq in seqs:
            time_seqs.append([float(t) for t in seq.split()])

    tmin = min([min(seq) for seq in time_seqs])
    tmax = max([max(seq) for seq in time_seqs])

    mark_seqs = []
    with open('./data/'+args.dataset+'_event.txt') as fmark:
        seqs = fmark.readlines()
        for seq in seqs:
            mark_seqs.append([int(k) for k in seq.split()])

    m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}

    evnt_seqs = [[((h_dt+time-tmin)*scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for time_seq, mark_seq in zip(time_seqs, mark_seqs)]
    #random.shuffle(evnt_seqs)

    return evnt_seqs, (0.0, ((tmax+t_dt)-(tmin-h_dt))*scale)


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)
    relu = nn.ReLU()
    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    
    TS, tspan = read_event_time(1.0, 1.0, 1.0)
    # TimeSequences, 全部事件(时间,事件类型)元组
    # tspn, 时间范围
    county_num = len(TS)
    # 维度
    dim_c, dim_h, dim_N, dt = 10, 10, county_num, 0.05
    # dim_c,dim_h: In order to better simulate the time series, the latent state z(t)∈ R^n is further split into 
    # two vectors: c(t)∈ R^n1 encodes the internal state, and h(t)∈ R^n2 encodes the memory of 
    # events up to time t, where n = n1 + n2.
    # dim_N,有多少种事件种类,这里最后的lambda函数,有多少事件种类就输出多少个lambda函数,如果设事件种类数量为1
    # dt, forward时间间隔
    nseqs = len(TS)

    TSTE = TS

    A_matrix = Variable(0.01*torch.ones((county_num, county_num)), requires_grad= True)
    #初始化A-matrix
    
    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=32, num_hidden=2, ortho=True, jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
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

            batch_id = np.arange(0,county_num,1)
            batch = [TS[seqid] for seqid in batch_id]

            # forward pass
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, batch, args.evnt_align, A_matrix,predict_first=False, rtol=1.0e-7, atol=1.0e-9)
            
            loss_meter.update(loss.item() / len(batch))

            # backward prop
            func.backtrace.clear()
            loss.backward()
            
            if torch.norm(A_matrix.grad)<1e-4:
                #终止条件
                for Ai in range(county_num):
                    for Aj in range(county_num):
                        with torch.no_grad():
                            if A_matrix[Ai,Aj]<=0:
                                A_matrix[Ai,Aj]=relu(A_matrix[Ai,Aj])+0.0001
                # 可以只在最后一步做此输出
                torch.save(A_matrix,"result.txt")
                break
            print(A_matrix)
            #np.savetxt(r'result.txt',A_matrix.detach().numpy(), fmt='%f', delimiter=',')
            
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, loss.item()/len(batch), loss_meter.avg, mete), flush=True)

            # step
            optimizer.step()

            it = it+1


    # computing testing error
    tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), tspan, dt, TS, args.evnt_align, predict_first = False, rtol=1.0e-7, atol=1.0e-9)
    visualize('graph_result', tsave, trace, lmbda, None, None, None, None, tsne, range(len(TS)), it, appendix="testing")
    print("iter: {:5d}, testing loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(it, loss.item()/len(TS), len(tsne)-len(TS), mete), flush=True)