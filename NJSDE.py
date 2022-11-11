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
import pickle
import os
import torch.optim as optim
from torch.autograd import Variable
from modules import RunningAverageMeter, ODEJumpFunc
from utils import forward_pass, visualize, read_timeseries
import matplotlib.pyplot as plt


signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

parser = argparse.ArgumentParser('stack_overflow')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--jump_type', type=str, default='read')
parser.add_argument('--paramr', type=str, default='params.pth')
parser.add_argument('--paramw', type=str, default='params.pth')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--nsave', type=int, default=1)
parser.add_argument('--fold', type=int, default=0)

#parser.add_argument('--dataset', type=str, default='n1')
#parser.add_argument('--dataset', type=str, default='new_n2')
#parser.add_argument('--dataset', type=str, default='LA_1x4_TTS')
#parser.add_argument('--dataset', type=str, default='California_Riverside_06065_2021-06-28_2021-12-28_15')
parser.add_argument('--dataset', type=str, default='California_Fresno_06019_2021-06-28_2021-12-28_12')
#parser.add_argument('--dataset', type=str, default='California_Riverside_06065_2021-06-28_2021-12-28_15')
#parser.add_argument('--dataset', type=str, default='California_Riverside_06065_2021-06-28_2021-12-28_15')
#parser.add_argument('--dataset', type=str, default='California_Riverside_06065_2021-06-28_2021-12-28_15')
#parser.add_argument('--dataset', type=str, default='California_Riverside_06065_2021-06-28_2021-12-28_15')


# small example data use 'm1' : Massachusetts from 20200928 to 20201228
# big example data use 'c1': california from 20200928 to 20201228

parser.set_defaults(restart=False, evnt_align=False, seed0=False, debug=False)
parser.add_argument('--restart', dest='restart', action='store_true')
parser.add_argument('--evnt_align', dest='evnt_align', action='store_true')
parser.add_argument('--seed0', dest='seed0', action='store_true')
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

outpath_para = 'para_save2'
outpath_graph = 'graph_result4'

loss_tot = []

def read_event_time2(scale=1.0, h_dt=0.0, t_dt=0.0):
    a_file = open('./data/'+args.dataset+'.pkl', "rb")
    data = pickle.load(a_file)
    
    time_seqs = []
    time_mark = {}
    tmp_time_seq = []
    for i in data.keys():           #遍历所有维度
        for j in data[i]:           #对于每个维度下的每个时间点
            tmp_time_seq.append(j)  #由于时间点已经处理过，不存在相同时间点
            time_mark[j] = i        #记录时间点j的所属维度i
    sort_tmp_time_seq = sorted(tmp_time_seq)    #从小到大排序时间点
    
    time_seqs.append(sort_tmp_time_seq)  
    
    tmin = min([min(seq) for seq in time_seqs]) #找出最小时间点
    tmax = max([max(seq) for seq in time_seqs]) #找出最大时间点
    
    mark_seqs=[]
    tmp_mark_seqs=[]
    
    for i in sort_tmp_time_seq:
        tmp_mark_seqs.append(time_mark[i])      #向标记序列中依次加入当前时间点的所属维度
    
    mark_seqs.append(tmp_mark_seqs)

    # marks to mark_id
    m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}

    # [[(t1_1,mk1_1),(t2_1,mk2_1)],[(t1_2,mk1_2),(t2_2,mk2_2)],[]]
    # 时间归一化，最小时间从h_dt=1.0开始
    evnt_seqs = [[((h_dt+time-tmin)*scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for time_seq, mark_seq in zip(time_seqs, mark_seqs)]
    
    return evnt_seqs, (0.0, ((tmax+t_dt)-(tmin-h_dt))*scale), len(data)



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
        
    # marks to mark_id
    m2mid = {m: mid for mid, m in enumerate(np.unique(sum(mark_seqs, [])))}
    
    

    # [[(t1_1,mk1_1),(t2_1,mk2_1)],[(t1_2,mk1_2),(t2_2,mk2_2)],[]]
    # 时间归一化，最小时间从h_dt=1.0开始
    evnt_seqs = [[((h_dt+time-tmin)*scale, m2mid[mark]) for time, mark in zip(time_seq, mark_seq)] for time_seq, mark_seq in zip(time_seqs, mark_seqs)]
    
    return evnt_seqs, (0.0, ((tmax+t_dt)-(tmin-h_dt))*scale),len(np.unique(mark_seqs))


if __name__ == '__main__':
    # write all parameters to output file
    print(args, flush=True)

    # fix seeding for randomness
    if args.seed0:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    # 维度 county_num
    
    # TimeSequences, 全部事件(时间,事件类型)元组
    # tspn, 时间范围, (min_t, max_t)
    # print(read_event_time(1.0, 1.0, 1.0))
    #TS, tspan,county_num = read_event_time(1.0, 1.0, 1.0)
    TS, tspan,county_num = read_event_time2(1.0, 1.0, 1.0)
    
    tot_pts = 0
    for i in range(len(TS)):
        tot_pts += len(TS[i])
    
    print(TS,county_num,tot_pts)
    #exit
    
    
    # dim_c,dim_h: In order to better simulate the time series, the latent state z(t)∈ R^n is further split into 
    # two vectors: c(t)∈ R^n1 encodes the internal state, and h(t)∈ R^n2 encodes the memory of 
    # events up to time t, where n = n1 + n2.
    # dim_N,有多少种事件种类,这里最后的lambda函数,有多少事件种类就输出多少个lambda函数
    # dt, forward时间间隔
    dim_c, dim_h, dim_N, dt = county_num, county_num, county_num, 0.05
    
    
    # initialize / load model
    func = ODEJumpFunc(dim_c, dim_h, dim_N, dim_N, dim_hidden=32, num_hidden=2, ortho=True, 
                       jump_type=args.jump_type, evnt_align=args.evnt_align, activation=nn.CELU())
    c0 = torch.randn(dim_c, requires_grad=True)
    h0 = torch.zeros(dim_h)
    it0 = 0
    
    #对微分方程的参数和A_matrix进行优化
    optimizer = optim.Adam([{'params': func.parameters()},
                            {'params': c0, 'lr': 1.0e-2},
                            ], lr=1e-3, weight_decay=1e-5)

    if args.restart:
        checkpoint = torch.load(args.paramr)
        func.load_state_dict(checkpoint['func_state_dict'])
        c0 = checkpoint['c0']
        h0 = checkpoint['h0']
        it0 = checkpoint['it0']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 设置计算loss移动平均值的函数
    loss_meter = RunningAverageMeter()

    # if read from history, then fit to maximize likelihood
    it = it0
    if func.jump_type == "read":
        while it < args.niters:
            func.jump_type = "read"
            # clear out gradients for variables
            #print(',<<<',c0.grad)
            optimizer.zero_grad()
            #print(',,,',c0.grad)
            
            # sample a mini-batch, create a grid based on that
            batch = TS
            
            # forward pass
            # z0: c0+h0
            '''
               utils.py——forward_pass——odeint_adjoint
            -> adjoint.py——odeint_adjoint——OdeintAdjointMethod——odeint
            -> odeint.py——solver.integrate
            -> solvers.py——AdaptiveStepsizeODESolver——integrate——advance
            -> adams.py——advance——VariableCoefficientAdamsBashforth——VariableCoefficientJumpAdamsBashforth——_adaptive_adams_step
            '''
            
            #print(func.W.state_dict().get('1.weight'))
            
            for i in func.W.state_dict().get('1.weight'):
                for j in range(len(i)):
                    if i[j]<0:
                        i[j]=0.00001
                
            #print(func.W.state_dict().get('1.weight'))
            
            tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), 
                        tspan, dt, batch, args.evnt_align, predict_first=False, rtol=1.0e-7, atol=1.0e-9)
            #tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), 
            #                            tspan, dt, batch, args.evnt_align, predict_first=False, rtol=1.0e-7, atol=1.0e-9)
            
            loss_meter.update(loss.item() / len(batch))

            # backward prop
            func.backtrace.clear()
            loss.backward()
            
            loss_tot.append(loss.item()/len(batch))
            #print(type(loss))
            #print(type(loss_tot))
            torch.save(loss_tot,"loss_curve_iter_"+str(it)+".txt")
            #exit
            
            print("iter: {}, current loss: {:10.4f}, running ave loss: {:10.4f}, type error: {}".format(it, 
                                                                        loss.item()/len(batch), loss_meter.avg, mete), flush=True)
            
            
            #print(func.W.state_dict().get('1.weight'))
            
            for i in func.W.state_dict().get('1.weight'):
                for j in range(len(i)):
                    if i[j]<0:
                        i[j]=0.00001
            
            tmp_sav = func.W.state_dict().get('1.weight')
            print(tmp_sav)
            
            if not os.path.exists('./Output_A/'+args.dataset):
                os.mkdir('./Output_A/'+args.dataset)
            
            #torch.save(torch.tensor(tmp_sav),'./Output_A/'+args.dataset+'/result_'+str(it)+'.npy')
            np.savetxt('./Output_A/'+args.dataset+'/result_'+str(it)+'.txt',tmp_sav, fmt='%f', delimiter=',')
            
            optimizer.step()

            it = it+1

            # validate and visualize
            if it % args.nsave == 0:
                # save
                print("iter for save")
                
                torch.save({'func_state_dict': func.state_dict(), 'c0': c0, 'h0': h0, 'it0': it, 
                            'optimizer_state_dict': optimizer.state_dict()}, outpath_para + '/' + '{:05d}'.format(it) + args.paramw)
                
                #'''
                tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), 
                                    tspan, dt, TS, args.evnt_align)
                
                # backward prop
                func.backtrace.clear()
                loss.backward()
                print("iter: {:5d}, validation loss: {:10.4f}, num_evnts: {:8d}, type error: {}".format(it, 
                                                        loss.item()/len(TS), len(tsne), mete), flush=True)

                
                # visualize
                tsave_ = torch.tensor([record[0] for record in reversed(func.backtrace)])
                trace_ = torch.stack(tuple(record[1] for record in reversed(func.backtrace)))
                visualize(outpath_graph, tsave, trace, lmbda, tsave_, trace_, None, None, tsne, 
                          range(len(TS)), it,appendix="validate")
                #
                
                func.jump_type="simulate"
                print("for simulate visual")
                #print(len(func.evnts))
                #print(type(func.evnts[0]))
                #print(func.evnts)
                tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, 
                             torch.cat((c0, h0), dim=-1), tspan, dt, [[]]*1, args.evnt_align)
                
                #print(type(tsave))
                
                # after simulate, you need to recreate tsave and tsne
                #tsave = 
                
                #print(len(func.evnts))
                simu_evnts = []
                for ti in range(len(func.evnts)):
                    tmp_tuple = []
                    for tj in range(len(func.evnts[ti])):
                        tmp_tuple.append(func.evnts[ti][tj].item())
                    simu_evnts.append(tuple(tmp_tuple))
                
                #simu_evnts = func.evnts
                #print(simu_evnts)
                #print(len(simu_evnts))
                #print(simu_evnts)
                
                evnts = [((evnt[0]),) + evnt[1:] for evnt in simu_evnts if tspan[0] < (evnt[0]) < tspan[1]]
                
                tgrid = np.round(np.arange(tspan[0], tspan[1]+dt, dt), decimals=8)
                
                tevnt = np.array([evnt[0] for evnt in evnts])
                
                tsave_s = np.sort(np.unique(np.concatenate((tgrid, tevnt))))
                
                #print(len(tsave_s))
                #print(tsave_s)
                
                t2tid = {t: tid for tid, t in enumerate(tsave_s)}

                gtid = [t2tid[t] for t in tgrid]
                
                tse = [(t2tid[evnt[0]],) + evnt[1:] for evnt in evnts]
                
                #print(len(tse))
                #print(tse)
                tsave_s = torch.tensor(tsave_s)
                
                visualize(outpath_graph, tsave, trace, lmbda, None, None, None, None,
                          tse, range(1), it, appendix="simulate",tsave_simu = tsave_s)
                #'''
    # simulate events
    func.jump_type="simulate"
    print("for simulate visual")
    tsave, trace, lmbda, gtid, tsne, loss, mete = forward_pass(func, torch.cat((c0, h0), dim=-1), 
                                                               tspan, dt, [[]]*1, args.evnt_align)
    visualize(outpath_graph, tsave, trace, lmbda, None, None, None, None, tsne, range(1), it, appendix="simulate")